"""
Lightweight frequency-domain residual blocks (DFFN-style amp/phase modulation).
Reference: Spatial-frequency dual-domain fusion (FFT on spatial feature maps, 1x1 conv on amp/phase).
"""
import torch
import torch.nn as nn


class FreqResidualBlock(nn.Module):
    """Single block: fft2 -> modulate amp and/or phase with 1x1 convs -> ifft2 -> returns (recon - x)."""

    def __init__(self, channels: int, fft_mode: str = "both", relu_slope: float = 0.2):
        super().__init__()
        assert fft_mode in ("amp", "phase", "both")
        self.fft_mode = fft_mode

        if fft_mode in ("amp", "both"):
            self.conv_amp_1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
            self.relu_amp = nn.LeakyReLU(relu_slope, inplace=False)
            self.conv_amp_2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if fft_mode in ("phase", "both"):
            self.conv_pha_1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
            self.relu_pha = nn.LeakyReLU(relu_slope, inplace=False)
            self.conv_pha_2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        if self.fft_mode == "amp":
            amp = self.conv_amp_2(self.relu_amp(self.conv_amp_1(amp)))
        elif self.fft_mode == "phase":
            phase = self.conv_pha_2(self.relu_pha(self.conv_pha_1(phase)))
        else:
            amp = self.conv_amp_2(self.relu_amp(self.conv_amp_1(amp)))
            phase = self.conv_pha_2(self.relu_pha(self.conv_pha_1(phase)))

        recon = torch.fft.ifft2(amp * torch.exp(1j * phase), dim=(-2, -1)).real
        return recon - x


class FreqResidualStack(nn.Module):
    """Stack of FreqResidualBlock; returns total delta (output - input) for residual injection."""

    def __init__(self, channels: int, fft_mode: str = "both", num_blocks: int = 1, relu_slope: float = 0.2):
        super().__init__()
        assert num_blocks >= 1
        self.blocks = nn.ModuleList(
            [FreqResidualBlock(channels, fft_mode, relu_slope) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x
        for blk in self.blocks:
            z = z + blk(z)
        return z - x
