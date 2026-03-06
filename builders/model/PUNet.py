#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright:    WZP
Filename:     PUNet.py
Description:

@author:      wuzhipeng
@email:       763008300@qq.com
@website:     https://wuzhipeng.cn/
@create on:   4/23/2021 7:46 PM
@software:    PyCharm
"""

__all__ = ["PUNet"]


import torch
import torch.nn as nn
from torchsummary import summary


class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DilatedBlock,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3,3), padding=(1,1),dilation=(1,1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3,3), padding=(2,2), dilation=(2,2)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3,3), padding=(3,3), dilation=(3,3)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(out_ch*3, out_ch, (3,3), padding=(1,1), dilation=(1,1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        o1 = self.layer1(x)
        o2 = self.layer2(x)
        o3 = self.layer3(x)
        o = torch.cat((o1,o2,o3),dim=1)
        o = self.relu(self.bn(x+self.conv(o)))
        return o

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ResidualBlock,self).__init__()
        self.conv = nn.Conv2d(out_ch, out_ch, (3,3), padding=(1,1), dilation=(1,1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.bn(x+self.conv(x)))

class PUNet(nn.Module):
    def __init__(self, num_channels=1):
        super(PUNet, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), padding=(1,1)),
            nn.ReLU(inplace=True),
        )

        self.dilated = nn.Sequential(
            *[DilatedBlock(64,64) for i in range(8)]
        )

        self.residual = nn.Sequential(
            *[ResidualBlock(64,64) for i in range(10)]
        )

        self.outc = nn.Conv2d(64, num_channels, (3,3), padding=(1,1))

    def forward(self, x):
        x = self.inc(x)
        x = self.dilated(x)
        x = self.residual(x)
        x = self.outc(x)

        return x


"""print layers and params of network"""
if __name__ == '__main__':
    from thop import profile, clever_format
    from thop.vision.basic_hooks import count_convNd, count_linear


    def calculate_model_metrics(model, input_tensor, device):
        params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        params_total = sum(p.numel() for p in model.parameters()) / 1e6
        flops, _ = profile(model, inputs=(input_tensor,))
        flops = flops / 1e9
        peak_memory = 0.0
        if torch.cuda.is_available():
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MiB

        return params_trainable, params_total, flops, peak_memory


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PUNet().to(device)
    input_tensor = torch.randn(1, 1, 256, 256).to(device)  # (B, C, H, W)

    params_trainable, params_total, flops, memory = calculate_model_metrics(model, input_tensor, device)
    print(f"{'Metric':<20} | {'Value':>15} | {'Unit':>10}")
    print("-" * 55)
    print(f"{'Trainable Params':<20} | {params_trainable:>15.2f} | {'M':>10}")
    print(f"{'Total Params':<20} | {params_total:>15.2f} | {'M':>10}")
    print(f"{'FLOPs':<20} | {flops:>15.4f} | {'GFLOPs':>10}")
    print(f"{'GPU Peak Memory':<20} | {memory:>15.2f} | {'MiB':>10}")

