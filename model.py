import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.res_scale = 0.3

    def forward(self, x):
        return x + self.res_scale * self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class DeblurResNet(nn.Module):
    """
    Encoder-decoder residual CNN. Downsampling 2x at two stages gives a
    receptive field ~4x larger than a flat full-res ResNet of the same
    depth (needed to actually "see" 15-35px blur kernels, not just denoise).
    Skip connections between matching encoder/decoder stages preserve detail.

    n_features=96, n_blocks=8 per stage -> ~14M params.
    """

    def __init__(self, in_channels=3, out_channels=3, n_features=96, n_blocks=8):
        super().__init__()

        # Encoder
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, n_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc0 = nn.Sequential(*[ResBlock(n_features) for _ in range(n_blocks)]) # Full

        self.down1 = DownBlock(n_features, n_features * 2)
        self.enc1 = nn.Sequential(*[ResBlock(n_features * 2) for _ in range(n_blocks)]) # 1/2

        self.down2 = DownBlock(n_features * 2, n_features * 4)
        self.enc2 = nn.Sequential(*[ResBlock(n_features * 4) for _ in range(n_blocks)]) # 1/4

        self.down3 = DownBlock(n_features * 4, n_features * 8)
        self.enc3 = nn.Sequential(*[ResBlock(n_features * 8) for _ in range(n_blocks)]) # 1/8 (New)

        # Decoder: 1/8 -> 1/4
        self.up3 = UpBlock(n_features * 8, n_features * 4)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(n_features * 8, n_features * 4, 3, padding=1),
            nn.GroupNorm(8, n_features * 4),
        )
        self.dec2 = nn.Sequential(*[ResBlock(n_features * 4) for _ in range(n_blocks // 2)])

        # Decoder: 1/4 -> 1/2
        self.up2 = UpBlock(n_features * 4, n_features * 2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features * 2, 3, padding=1),
            nn.GroupNorm(8, n_features * 2),
        )
        self.dec1 = nn.Sequential(*[ResBlock(n_features * 2) for _ in range(n_blocks // 2)])

        # Decoder: 1/2 -> full
        self.up1 = UpBlock(n_features * 2, n_features)
        self.fuse0 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features, 3, padding=1),
            nn.GroupNorm(8, n_features),
        )
        self.dec0 = nn.Sequential(*[ResBlock(n_features) for _ in range(n_blocks // 2)])

        # Tail (Fixed Tanh & out_scale bugs)
        self.tail = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features, out_channels, 3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Zero-init the last layer to guarantee training stability
                nn.init.zeros_(self.tail[-1].weight)
                if self.tail[-1].bias is not None:
                    nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        f0 = self.enc0(self.head(x))           
        f1 = self.enc1(self.down1(f0))         
        f2 = self.enc2(self.down2(f1))         
        f3 = self.enc3(self.down3(f2)) # 1/8 res

        d2 = self.up3(f3) # 1/8 -> 1/4                      
        d2 = self.fuse2(torch.cat([d2, f2], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up2(d2) # 1/4 -> 1/2                      
        d1 = self.fuse1(torch.cat([d1, f1], dim=1))
        d1 = self.dec1(d1)

        d0 = self.up1(d1) # 1/2 -> full                       
        d0 = self.fuse0(torch.cat([d0, f0], dim=1))
        d0 = self.dec0(d0)

        return torch.clamp(x + self.tail(d0), 0.0, 1.0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = DeblurResNet()
    x = torch.rand(2, 3, 256, 256)
    y = m(x)
    print("Params:", m.count_parameters())
    print("Output:", y.shape, y.min().item(), y.max().item())