import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from engine.extre_module.ultralytics_nn.conv import Conv, DWConv, Concat

from engine.extre_module.ultralytics_nn.block import C2f

from einops import rearrange, reduce


__all__=["TDMSA"]


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor


class TDMSA(nn.Module):
    def __init__(self, dim, num_heads=8, atten_drop=0., proj_drop=0.,
                 dilation=[3, 5, 7], fc_ratio=4, pool_ratio=16, bias=True):
        super(TDMSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 两个分支模块
        self.MSC = MutilScal(dim = dim // 2, fc_ratio = fc_ratio, dilation = dilation, pool_ratio = pool_ratio)
        self.FRE = Freq_attention(dim =dim // 2, num_heads = num_heads, bias = bias)
        # 融合卷积
        self.kv = Conv(dim, 2 * dim, 1)

        # 通道注意
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, 1),
            nn.ReLU6(),
            nn.Conv2d(dim // fc_ratio, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape

        # --- 通道划分 ---
        x1, x2 = x.chunk(2, dim=1)

        # --- 两种分支 ---
        kv1 = self.MSC(x1)  # 多尺度空间分支
        kv2 = self.FRE(x2)  # 频域增强分支

        # --- 尺寸对齐 ---
        Ht = min(kv1.shape[2], kv2.shape[2])
        Wt = min(kv1.shape[3], kv2.shape[3])
        kv1 = F.interpolate(kv1, size=(Ht, Wt), mode='bilinear', align_corners=False)
        kv2 = F.interpolate(kv2, size=(Ht, Wt), mode='bilinear', align_corners=False)

        # --- 拼接 ---
        kv = torch.cat([kv1, kv2], dim=1)

        # --- 后续注意力 ---
        kv = self.kv(kv)

        q = rearrange(x, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.num_heads)
        k, v = rearrange(kv, 'b (kv h d) h1 w1 -> kv b h (h1 w1) d', kv=2, h=self.num_heads)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, 'b h (hh ww) d -> b (h d) hh ww', h=self.num_heads, hh=H, ww=W)

        c_attn = self.fc(self.avgpool(x)) * u
        return attn + c_attn

class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = Conv(dim, dim//fc_ratio)
        self.conv0_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-3], g=dim//fc_ratio)
        self.conv0_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv1_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv2_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-1], g=dim//fc_ratio)
        self.conv2_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv3 = Conv(dim, dim, 1)

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)
        self.Fusing = MutualGuidedFusionModel([dim, dim], dim)


    def forward(self, x):
        u = x.clone()

        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1+attn0_2)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1+attn1_2)
        attn2_3 = self.conv2_3(attn2_2)
        attn = self.Fusing([attn1_3,attn2_3])+self.Fusing([attn1_3,attn0_3])

        # attn = attn0_3 + attn1_3 + attn2_3
        attn = self.conv3(attn)
        attn = attn * u

        pool = self.Avg(attn)
        return pool

class Freq_attention(nn.Module):
    def __init__(self, dim, num_heads, bias, ):
        super(Freq_attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=True),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim, 1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.shape

        q_f = torch.fft.fft2(x.float())
        k_f = torch.fft.fft2(x.float())
        v_f = torch.fft.fft2(x.float())

        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_f = torch.nn.functional.normalize(q_f, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)

        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = custom_complex_normalization(attn_f, dim=-1)
        out_f = torch.abs(torch.fft.ifft2(attn_f @ v_f))
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_f_l = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))
        out = self.project_out(torch.cat((out_f, out_f_l), 1))

        return out


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MutualGuidedFusionModel(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)

        self.se = SEAttention(inc[1] * 2)

        if (inc[1] * 2) != ouc:
            self.conv1x1 = Conv(inc[1] * 2, ouc)
        else:
            self.conv1x1 = nn.Identity()

    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)

        x_concat = torch.cat([x0, x1], dim=1)  # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return self.conv1x1(torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1))

class GConv3(nn.Module):
    def __init__(self, in_features, out_features,hidden_features=None,
                 act_layer=nn.GELU, drop=0., n=1) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features =  in_features
        hidden_features = int(2 * hidden_features / 3)

        # 1x1 卷积，生成 2 * hidden_features 通道
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        # 深度卷积
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=hidden_features),
            act_layer()
        )

        # 1x1 卷积恢复到 out_features
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop)

        # 如果 in_features 和 out_features 不一样，就加 shortcut 卷积保证能相加
        if in_features != out_features:
            self.shortcut = nn.Conv2d(in_features, out_features, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x_shortcut = self.shortcut(x)  # 通道数对齐
        x, v = self.fc1(x).chunk(2, dim=1)  # 分块
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x

class GConv4(nn.Module):
    def __init__(self, in_features, out_features,hidden_features=None,
                 act_layer=nn.GELU, drop=0., n=1) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features =  in_features
        hidden_features = int(2 * hidden_features / 3)

        # 1x1 卷积，生成 2 * hidden_features 通道
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        # 深度卷积
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=hidden_features),
            act_layer()
        )

        # 1x1 卷积恢复到 out_features
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop)

        # 如果 in_features 和 out_features 不一样，就加 shortcut 卷积保证能相加
        if in_features != out_features:
            self.shortcut = nn.Conv2d(in_features, out_features, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):

        x_shortcut = self.shortcut(x)  # 通道数对齐
        x, v = self.fc1(x).chunk(2, dim=1)  # 分块
        x = self.dwconv(x) * v
        x1 = x
        x2 = x
        x2 = x2 + v
        x2 = torch.sigmoid(x2)
        x = x1 * x2
        x = self.drop(x)
        x = self.fc2(x)

        x = self.drop(x)

        return x_shortcut + x
