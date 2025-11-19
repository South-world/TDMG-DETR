import os, sys

from mmcv.cnn import Conv2d

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings  
warnings.filterwarnings('ignore')  
from calflops import calculate_flops 
  
import copy  
from collections import OrderedDict

import torch   
import torch.nn as nn  
import torch.nn.functional as F 

from engine.core import register
from engine.extre_module.ultralytics_nn.conv import Conv, autopad
from engine.extre_module.ultralytics_nn.block import C2f
from engine.extre_module.custom_nn.attention.simam import SimAM


__all__ = ['MFF']

class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand     
        super().__init__() 
        self.c = c2 // 2  
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)     
 
    def forward(self, x):  
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)     
        x2 = self.cv2(x2)  
        return torch.cat((x1, x2), 1)

class MSGconv(nn.Module):
    def __init__(self, in_features, out_features,hidden_features=None,
                 act_layer=nn.GELU, drop=0., n=1) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        # 深度卷积
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=hidden_features),
            act_layer()
        )
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features//3 , 3, 1, 1, groups=hidden_features//3 )
        self.dwconv5 = nn.Conv2d(hidden_features, hidden_features//3 , 5, 1, 2, groups=hidden_features//3 )
        self.dwconv7 = nn.Conv2d(hidden_features, hidden_features//3 , 7, 1, 3, groups=hidden_features//3 )
        self.act = nn.Sequential(
            act_layer()
        )
        self.act2 = SimAM()

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop)

        if in_features != out_features:
            self.shortcut = nn.Conv2d(in_features, out_features, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x_shortcut = self.shortcut(x)  # 通道数对齐
        x, v = self.fc1(x).chunk(2, dim=1)  # 分块
        x = torch.cat([self.dwconv3(x), self.dwconv5(x), self.dwconv7(x)], dim=1)
        x = self.act(x)
        x = x * self.act2(v)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x
 
class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()     
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential( 
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)     
        )     
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()   
        self.conv3 = ADown(inc[2], hidc)
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)     
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = Conv(hidc * 3, int(hidc / e))     
    
    def forward(self, x):     
        x1, x2, x3 = x  
        x1 = self.conv1(x1)
        x2 = self.conv2(x2) 
        x3 = self.conv3(x3)
  
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)

        x = x + feature
        return self.conv_1x1(x)

class MGDF(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)
        # print("inc[1]", inc[1])

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)

        self.dw_conv = nn.ModuleList(
            nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = MSGconv(hidc * 3, int(hidc / e))

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        # print("DEBUG:FEATURE.shape:",x.shape)

        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)

        x = x + feature
        return self.conv_1x1(x)


@register(force=True)
class MFF(nn.Module):
    def __init__(self, 
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 mff_ks=[3, 5, 7, 9],
                 depth_mult=1.0,
                 out_strides=[8, 16, 32],
                 eval_spatial_size=None,
                 ):   
        super().__init__()  
        from engine.deim.hybrid_encoder import TransformerEncoderLayer, TransformerEncoder # 避免 circular import  
    
        # 保存传入的参数为类的成员变量
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = out_strides

        assert len(in_channels) == 3
   
        # 输入投影层：将不同通道数的输入特征图投影到统一的 hidden_dim
        self.input_proj = nn.ModuleList()    
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim))
            ]))   
            self.input_proj.append(proj)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )
        # 为每个指定层创建独立的 Transformer 编码器     
        self.encoder = nn.ModuleList([ 
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])    

        # --------------------------- 第一阶段     
        self.FocusFeature_1 = MGDF(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=mff_ks)
   
        self.p4_to_p5_down1 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True) 

        self.p4_to_p3_up1 = nn.Upsample(scale_factor=2)     
        self.p3_block1 = C2f(hidden_dim * 2, hidden_dim, round(3 * depth_mult), shortcut=True)
    
        # --------------------------- 第二阶段
        self.FocusFeature_2 = MGDF(inc=[hidden_dim, hidden_dim, hidden_dim], kernel_sizes=mff_ks)

        self.p4_to_p5_down2 = Conv(hidden_dim, hidden_dim, k=3, s=2)
        self.p5_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)
     
        if len(out_strides) == 3:    
            self.p4_to_p3_up2 = nn.Upsample(scale_factor=2)
            self.p3_block2 = C2f(hidden_dim * 3, hidden_dim, round(3 * depth_mult), shortcut=True)
    
        # 初始化参数，包括预计算位置编码     
        self._reset_parameters()  

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]

                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod  
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """ 
        生成 2D sine-cosine 位置编码
        Args:
            w (int): 特征图宽度    
            h (int): 特征图高度
            embed_dim (int): 嵌入维度，必须能被 4 整除
            temperature (float): 温度参数，控制频率     
        Returns:
            torch.Tensor: 位置编码张量，形状为 [1, w*h, embed_dim]
        """

        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')  # 生成 2D 网格
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'     
        pos_dim = embed_dim // 4
        # 计算频率因子  
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]  # [w*h, pos_dim]
        out_h = grid_h.flatten()[..., None] @ omega[None]  # [w*h, pos_dim] 

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):   
        """
        前向传播函数  
        Args:
            feats (list[torch.Tensor]): 输入特征图列表，形状为 [B, C, H, W]，长度需与 in_channels 一致
        Returns:
            list[torch.Tensor]: 融合后的多尺度特征图列表
        """   

        assert len(feats) == len(self.in_channels)
 
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
    
        # Transformer 编码器：对指定层进行特征增强
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):  
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:    
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                # Transformer 编码器处理
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
    
        fouce_feature1 = self.FocusFeature_1(proj_feats[::-1]) # 倒序是因为FocusFeature要求从小特征图到大特征图输入
     
        fouce_feature1_to_p5_1 = self.p4_to_p5_down1(fouce_feature1) # fouce_feature1 to p5   
        fouce_feature1_to_p5_2 = self.p5_block1(torch.cat([fouce_feature1_to_p5_1, proj_feats[2]], dim=1))     

        fouce_feature1_to_p3_1 = self.p4_to_p3_up1(fouce_feature1) # fouce_feature1 to p3   
        fouce_feature1_to_p3_2 = self.p3_block1(torch.cat([fouce_feature1_to_p3_1, proj_feats[0]], dim=1)) 

        fouce_feature2 = self.FocusFeature_2([fouce_feature1_to_p5_2, fouce_feature1, fouce_feature1_to_p3_2])

        fouce_feature2_to_p5 = self.p4_to_p5_down2(fouce_feature2) # fouce_feature2 to p5    
        fouce_feature2_to_p5 = self.p5_block2(torch.cat([fouce_feature2_to_p5, fouce_feature1_to_p5_1, fouce_feature1_to_p5_2], dim=1))   

        if len(self.out_strides) == 3:
            fouce_feature2_to_p3 = self.p4_to_p3_up2(fouce_feature2) # fouce_feature2 to p3
            fouce_feature2_to_p3 = self.p3_block2(torch.cat([fouce_feature2_to_p3, fouce_feature1_to_p3_1, fouce_feature1_to_p3_2], dim=1))
            return [fouce_feature2_to_p3, fouce_feature2, fouce_feature2_to_p5]     
        else:   
            return [fouce_feature2, fouce_feature2_to_p5]
