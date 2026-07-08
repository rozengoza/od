"""
CBAM (Convolutional Block Attention Module) - Woo et al., ECCV 2018.

This is the novel-method contribution of the project: a lightweight
channel + spatial attention block inserted right before each of YOLOv8's
three detection heads (P3/P4/P5). Helmets and bare heads are small objects
that often sit in cluttered backgrounds (traffic, construction sites); CBAM
re-weights feature maps to emphasize the channels/regions that matter for
those small objects instead of dominant background texture, before the
final prediction convolution sees them.

Channel/spatial dimensions are inferred lazily on first forward pass, so
the module can be dropped into the YOLOv8 YAML (models/yolov8s-cbam.yaml)
without hardcoding channel counts per model scale (n/s/m/l/x).
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, ratio: int = 16):
        super().__init__()
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = None

    def _build(self, channels: int, device, dtype):
        hidden = max(channels // self.ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        ).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            self._build(x.shape[1], x.device, x.dtype)
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention. Preserves input shape,
    so it can be inserted anywhere in a YOLOv8 YAML without affecting
    downstream channel-count bookkeeping."""

    def __init__(self, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
