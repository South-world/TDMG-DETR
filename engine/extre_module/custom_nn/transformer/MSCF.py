import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from calflops import calculate_flops

from engine.extre_module.ultralytics_nn.conv import Conv,DWConv
from engine.extre_module.ultralytics_nn.block import C2f

from einops import rearrange, reduce
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from timm.layers import DropPath
from engine.deim.hybrid_encoder import TransformerEncoderBlock

__all__ = ['MSCF']



class GConv(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)  # chunk函数用于将张量按指定维度分割成多个等大小或不等大小的块。当元素数量不能整除指定的块数时，最后一个块的大小会较小
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


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
        x1 = x
        x2 = x
        x2 = x2 + v
        x2 = torch.sigmoid(x2)
        x = x1 * x2
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
        x1 = self.dwconv(x1)
        x2 = x
        x2 = x2 + v
        x2 = torch.sigmoid(x2)
        x = x1 * x2
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class MSCF(nn.Module):
    def __init__(self, dim, nheads=8, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(MSCF, self).__init__()
        assert dim % nheads == 0, f"dim {dim} should be divided by num_heads {nheads}."
        self.dim = dim
        self.num_heads = nheads
        head_dim = dim // nheads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, src, src_mask=None, pos_embed=None):
        x=src
        u = x.clone()
        B, C, H, W = x.shape

        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
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

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.conv3(attn)
        attn = attn * u

        pool = self.Avg(attn)

        return pool



if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = GConv3(64, 32,32).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)