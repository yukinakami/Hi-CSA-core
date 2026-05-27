import torch
import torch.nn as nn

class TimeSeriesInstanceNorm(nn.Module):
    """
    Instance Normalization for Time Series
    Input shape: [B, C, T]
    Normalizes each sample's channel independently.
    """
    def __init__(self, num_channels, affine=True):
        super(TimeSeriesInstanceNorm, self).__init__()
        # InstanceNorm1d works on [B, C, T]
        self.inst_norm = nn.InstanceNorm1d(num_channels, affine=affine)
    
    def forward(self, x):
        """
        x: [B, C, T] tensor
        """
        return self.inst_norm(x)