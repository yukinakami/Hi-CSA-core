import torch
import torch.nn as nn
import torch.fft as fft
from einops import repeat, rearrange, reduce
import math

class ScalePredictor(nn.Module):
    def __init__(self, in_channels):
        super(ScalePredictor, self).__init__()
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.fourier = Fourier(pred_len=0, k = 3, low_freq = 1)
        self.scale_learner_sample = ScaleLearningSimple(input_dim = 3 * in_channels, k = 3)
        self.scale_learning_gmm = ScaleLearningGMM(input_dim = 3 * in_channels, k = 3)

        
        
    
    def forward(self, x, fourier_method = False):

        x = self.norm(x)
        
        if fourier_method == True:
        
            fourier_result, _, _, _ = self.fourier(x)
            high_frequence = x - fourier_result
            high_frequence = high_frequence.mean(dim=1)
            fourier_result = fourier_result.mean(dim=1)
            x_features = x.mean(dim=1)
            x_representation = torch.cat([x_features, high_frequence, fourier_result], dim=-1)
            scale = self.scale_learner_sample(x_representation)

            return scale
        
        fourier_result, _, _, _ = self.fourier(x)
        high_frequence = x - fourier_result
        high_frequence = high_frequence.mean(dim=1)
        fourier_result = fourier_result.mean(dim=1)
        x_features = x.mean(dim=1)
        x_representation = torch.cat([x_features, high_frequence, fourier_result], dim=-1)
        scale = self.scale_learning_gmm(x_representation)
        
        return scale


class Fourier(nn.Module):
    def __init__(self, pred_len, k, low_freq=1):
        super().__init__()
        self.pred_len = pred_len
        self.f = k
        self.low_freq = low_freq


    def forward(self, x):

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0: # if the t(length of sequence) is even number
            x_freq = x_freq[:, self.low_freq:-1] #decline lowest freqence
            f = fft.rfftfreq(t)[self.low_freq:-1]
        
        else:
            x_freq = x_freq[: self.low_freq:] #odd number
            f = fft.rfftfreq(t)[self.low_freq:]

            x_freq, index_tuple = self.topk(x_freq)
            f = repeat(f, 'f -> b f c', b = x_freq.size(0), c = x_freq.size(2))
            f = f.to(x_freq.device)
            f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
            # output (B, k, 1, C)

        x_freq, f, amplitude, phase = self.Frequency_Extrapolatio(x_freq, f, t)

        fourier_result = self.IDFT(x_freq, t, f, amplitude, phase)

        return fourier_result, f, amplitude, phase

    def topk(self, x_freq):

        value, index = torch.topk(x_freq.abs(), self.f, dim=1)
        batch_index, channel_index = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        #torch.arange(): generate a sequence of step=1
        #a, b = torch.meshgrid(a,b): generate a:(a,b1),(a,b2),...,(a,bn) b:(a1,b),(a2,b),...,(an,b)
        index_tuple = (batch_index.unsqueeze(1), index, channel_index.unsqueeze(1))
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
        phase = rearrange(torch.angle(x_freq) / t, 'b f d -> b f () d')
        
        return x_freq, f, amplitude, phase
    
    def IDFT(self, x_freq, t, f, amplitude, phase):

        timeline_total = torch.arange(t + self.pred_len, dtype=torch.float)
        timeline_redio = rearrange(timeline_total, 't -> () () t ()').to(x_freq.device)
        # make sure timeline_redio and x_freq in the same device

        #IDFT equation
        x_wo_noise = amplitude * torch.cos(2 * math.pi * f * timeline_redio + phase)

        x_time = reduce(x_wo_noise, 'b f t d -> b t d', 'sum')
        return x_time
    
class ScaleLearningSimple(nn.Module):
    def __init__(self, input_dim, k = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k)
        )

    def forward(self, x):

        scale = torch.nn.functional.softplus(self.mlp(x))
        return scale

class ScaleLearningGMM(nn.Module):
    def __init__(self, input_dim, k = 3):
        super.__init__()
        self.k = k
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k * 3) #three distribution
        )

    def foward(self, x):
        scale_distribution = self.fc(x.size(0)) #[B, 3k]

        mu, log_sigma, weight_logits = torch.chunk(scale_distribution, 3, dim = -1)
        mu = nn.functional.softplus(mu)
        sigma = nn.functional.softplus(log_sigma)
        weights = nn.functional.softplus(weight_logits, dim = -1)

        # reparameterization sampling
        eps = torch.randn_like(mu) # generate a tensor with the same tensor as mu, which follow the standard normal distribution
        scales = mu + sigma * eps # generate a distribution scales follow the distribution of scale_distribution
        scales = torch.clamp(scales, min=1e-3)

        return scales, weights, mu, sigma