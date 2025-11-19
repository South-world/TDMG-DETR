import torch.nn as nn
import torch
from mmcv.ops import DeformConv2d
import torch.nn.functional as F

class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # 公式：hardsigmoid(x) = clip(x * 0.2 + 0.5, 0, 1)
        return torch.clamp(x * 0.2 + 0.5, min=0.0, max=1.0)


class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = DeformConv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = HardSigmoid()
        self.offset = nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = self.offset(out)
        result = x * out
        return result

class AdaptiveGlobalFilter(nn.Module):
    def __init__(self, ratio=10, dim=32, base_hw=512):
        super().__init__()
        self.ratio = ratio
        self.filter = nn.Parameter(torch.randn(dim, base_hw, base_hw, 2, dtype=torch.float32))
        self.alhpa = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        b, c, h, w = x.shape
        crow, ccol = h // 2, w // 2

        # 动态生成掩码
        mask_low = torch.zeros((h, w), device=x.device)
        mask_high = torch.ones((h, w), device=x.device)
        mask_low[crow - self.ratio:crow + self.ratio, ccol - self.ratio:ccol + self.ratio] = 1
        mask_high[crow - self.ratio:crow + self.ratio, ccol - self.ratio:ccol + self.ratio] = 0
        mask_low, mask_high = mask_low[None, None, :, :], mask_high[None, None, :, :]

        # 频域变换
        x_fre = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm='ortho'))

        # 调整 filter 尺寸
        weight = torch.view_as_complex(self.filter)
        weight = F.interpolate(weight.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        # 低频/高频滤波
        x_fre_low = x_fre * mask_low
        x_fre_high = x_fre * mask_high
        x_fre_low = x_fre_low * weight.unsqueeze(0)
        x_fre_new = self.alhpa * x_fre_low + (1-self.alhpa) * x_fre_high

        # 逆变换
        x_out = torch.fft.ifft2(torch.fft.ifftshift(x_fre_new, dim=(-2, -1))).real
        return x_out

