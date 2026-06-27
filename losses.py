import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        for p in vgg.parameters(): 
            p.requires_grad = False
        self.vgg = vgg
        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n = (pred - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps2).mean()


def _gaussian_kernel(window_size, sigma, channels):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def _ssim(x, y, window, window_size, channels, C1=0.01 ** 2, C2=0.03 ** 2):
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", _gaussian_kernel(window_size, sigma, channels))

    def forward(self, pred, target):
        return 1.0 - _ssim(pred, target, self.window, self.window_size, self.channels)


class LaplacianLoss(nn.Module):
    """Penalizes missing high-frequency detail/edges directly -
    fixes PSNR/SSIM being fooled by smooth-but-blurry outputs."""

    def __init__(self, channels=3):
        super().__init__()
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        k = k.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3).contiguous()
        self.register_buffer("kernel", k)
        self.channels = channels

    def forward(self, pred, target):
        edge_p = F.conv2d(pred, self.kernel, padding=1, groups=self.channels)
        edge_t = F.conv2d(target, self.kernel, padding=1, groups=self.channels)
        return F.l1_loss(edge_p, edge_t)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.1, gamma=0.2, eta=0.3, channels=3):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eta = alpha, beta, gamma, eta
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss(channels=channels)
        self.lap = LaplacianLoss(channels=channels)
        self.vgg = PerceptualLoss() # New

    def forward(self, pred, target):
        return (self.alpha * self.charb(pred, target)
                + self.beta * self.ssim(pred, target)
                + self.gamma * self.lap(pred, target)
                + self.eta * self.vgg(pred, target))


@torch.no_grad()
def psnr_batch(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3]).clamp(min=1e-10)
    return (10.0 * torch.log10(max_val ** 2 / mse)).mean().item()


@torch.no_grad()
def ssim_batch(pred, target, window_size=11, sigma=1.5):
    c = pred.shape[1]
    window = _gaussian_kernel(window_size, sigma, c).to(pred.device)
    return _ssim(pred, target, window, window_size, c).item()