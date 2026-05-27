import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
import math

class CScalePredictor(nn.Module):
    def __init__(self, in_channels, pred_len, flourier_k, gmm_k):
        super(CScalePredictor, self).__init__()
        #self.pred_len = pred_len

        self.norm = nn.InstanceNorm1d(in_channels, affine=False)
        self.fourier = Fourier(pred_len, flourier_k, low_freq = 1)
        self.scale_learner_sample = ScaleLearningSimple(flourier_k, input_dim = 3 * in_channels)
        self.scale_learning_gmm = ScaleLearningGMM(gmm_k, input_dim = 8 * flourier_k)
       
    
    def forward(self, x):

        # FFT amplitude/phase are signal descriptors for the scale distribution.
        # Detaching this path avoids undefined angle/log gradients near zero-magnitude bins.
        with torch.no_grad():
            x_perm = x.permute(0, 2, 1)
            x_norm = self.norm(x_perm)
            x_out = x_norm.permute(0, 2, 1)
            _, f, apm, phase = self.fourier.components(x_out)
  
        mu, sigma, logits = self.scale_learning_gmm(f, apm, phase)
        
        return mu, sigma, logits


class Fourier(nn.Module):
    def __init__(self, pred_len, flourier_k, low_freq=1):
        super().__init__()
        self.pred_len = pred_len
        self.f = flourier_k
        self.low_freq = low_freq


    def forward(self, x):

        x_freq, f, amplitude, phase = self.components(x)
        fourier_result = self.IDFT(x_freq, x.shape[1], f, amplitude, phase)

        return fourier_result, f, amplitude, phase

    def components(self, x):

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0: # if the t(length of sequence) is even number
            x_freq = x_freq[:, self.low_freq:-1] #decline lowest freqence
            f = fft.rfftfreq(t)[self.low_freq:-1]
        
        else:
            x_freq = x_freq[:, self.low_freq:] #odd number
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk(x_freq)
        f = repeat(f, 'f -> b f c', b = x_freq.size(0), c = x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        # output (B, k, 1, C)

        x_freq, f, amplitude, phase = self.Frequency_Extrapolatio(x_freq, f, t)

        return x_freq, f, amplitude, phase

    def topk(self, x_freq):

        k = min(self.f, x_freq.size(1))
        if k <= 0:
            raise ValueError("No valid frequency bins. Increase sequence length or lower low_freq.")

        _, index = torch.topk(x_freq.abs(), k, dim=1)
        batch_index = torch.arange(x_freq.size(0), device=x_freq.device).view(-1, 1, 1)
        channel_index = torch.arange(x_freq.size(2), device=x_freq.device).view(1, 1, -1)
        index_tuple = (batch_index, index, channel_index)
        #[B,1,C] + [B,k,C] + [B,1,C] → [B,k,C], get individual topk B/C
        #The top-k frequency for each channel in each batch
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple
    
    def Frequency_Extrapolatio(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1) # Complex Conjugate, a+bi -> a-bi
        # cancel the imaginary part
        f = torch.cat([f, -f], dim=1)
        #timeline_total = torch.arange(t + self.pred_len, dtype=torch.float)
        #timeline_redio = rearrange(timeline_total, 't -> () () t ()').to(x_freq.device)
        # make sure timeline_redio and x_freq in the same device

        # Amplitude, /t is to normalize
        amplitude = rearrange(torch.abs(x_freq) / t, 'b f d -> b f () d')
        # Phase
        phase = rearrange(torch.angle(x_freq), 'b f d -> b f () d')
        
        return x_freq, f, amplitude, phase
    
    def IDFT(self, x_freq, t, f, amplitude, phase):

        timeline_total = torch.arange(
            t + self.pred_len,
            dtype=amplitude.dtype,
            device=x_freq.device
        )
        timeline_redio = rearrange(timeline_total, 't -> () () t ()')
        # make sure timeline_redio and x_freq in the same device

        #IDFT equation
        x_wo_noise = amplitude * torch.cos(2 * math.pi * f * timeline_redio + phase)
        x_wo_noise = torch.nan_to_num(x_wo_noise, nan=0.0, posinf=1e4, neginf=-1e4)

        x_time = reduce(x_wo_noise, 'b f t d -> b t d', 'sum')
        x_time = torch.nan_to_num(x_time, nan=0.0, posinf=1e4, neginf=-1e4)
        return x_time, timeline_redio
    
class ScaleLearningSimple(nn.Module):
    def __init__(self, flourier_k, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, flourier_k)
        )

    def forward(self, x):

        scale = torch.nn.functional.softplus(self.mlp(x))
        return scale

class ScaleLearningGMM(nn.Module):
    def __init__(self, gmm_k, input_dim):
        super().__init__()
        self.gmm_k = gmm_k
        #self.k = gmm_k
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64) 
        )

        self.mu_head = nn.Linear(64, gmm_k)
        self.sigma_head = nn.Linear(64, gmm_k)
        self.weight_head = nn.Linear(64, gmm_k)

    def forward(self, f, ampltude, phase):
        
        #B, F, _, D = f.shape
        
        features = self.FrequencyComponent(f, ampltude, phase)

        features = torch.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
        features = torch.clamp(features, -20.0, 20.0)

        h = self.mlp(features)
        h = torch.nan_to_num(h, nan=0.0, posinf=20.0, neginf=-20.0)
        h = torch.clamp(h, -20.0, 20.0)

        mu = F.softplus(self.mu_head(h)) + 1e-4
        mu = torch.clamp(mu, min=1e-4, max=128.0)
        log_sigma = torch.clamp(self.sigma_head(h), -5.0, 5.0)
        logits = torch.clamp(self.weight_head(h), -10.0, 10.0)

        sigma = F.softplus(log_sigma) + 1e-4
        sigma = torch.clamp(sigma, min=1e-2, max=64.0)
        weights = torch.softmax(logits, dim = -1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        
        # mu = mu.reshape(B, D, self.gmm_k)
        # sigma = sigma.reshape(B, D, self.gmm_k)
        # logits = logits.reshape(B, D, self.gmm_k)

        return mu, sigma, weights
    
    def FrequencyComponent(self, f, amplitude, phase):

        '''f, amplitude, phase = (b, f, 1, d)'''

        phase_sin = torch.sin(phase).unsqueeze(-1) #(b, f, 1, d, 1)
        phase_cos = torch.cos(phase).unsqueeze(-1)

        amplitude = torch.nan_to_num(amplitude, nan=0.0, posinf=1e6, neginf=0.0)
        amp = torch.log(torch.clamp(amplitude, min=1e-6, max=1e6)).unsqueeze(-1)

        f = f.unsqueeze(-1)

        features = torch.cat([phase_sin, phase_cos, amp, f], dim = -1) #(b, f, 1, d, 4)
        features = reduce(features, 'b f t d k -> b f d k', 'sum')

        features = rearrange(features, 'b f d k -> b d (f k)')
        feature_dim = features.shape[-1]

        if feature_dim < self.input_dim:
            features = F.pad(features, (0, self.input_dim - feature_dim))
        elif feature_dim > self.input_dim:
            features = features[..., :self.input_dim]

        return features #(b, d, 4 * freq)


