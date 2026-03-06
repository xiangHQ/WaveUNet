
__all__ = ["DLPUNet"]

import torch
import torch.nn as nn
import random
import numpy as np
# import matplotlib.pyplot as plt
from torchsummary import summary

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # работает медленнее, но зато воспроизводимость!


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False, padding_mode='replicate')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class ResidualBlockUp(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockUp, self).__init__()

        self.conv1 = conv3x3(in_channels, 2 * out_channels, stride)
        self.bn1 = nn.BatchNorm2d(2 * out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(2 * out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.bn1(out)
        out = self.relu(out)
        out = out + residual
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class DLPUNet(torch.nn.Module):

    def __init__(self):
        super(DLPUNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = ResidualBlock(1, 8)
        self.block2 = ResidualBlock(8, 16)
        self.block3 = ResidualBlock(16, 32)
        self.block4 = ResidualBlock(32, 64)
        self.block5 = ResidualBlock(64, 128)
        self.block6 = ResidualBlock(128, 256)

        self.block_up1 = ResidualBlockUp(256, 64)
        self.block_up2 = ResidualBlockUp(128, 32)
        self.block_up3 = ResidualBlockUp(64, 16)
        self.block_up4 = ResidualBlockUp(32, 8)
        self.block_up5 = ResidualBlockUp(16, 1)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=2,
            stride=2)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=2,
            stride=2)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=2,
            stride=2)

        self.up_trans_5 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=8,
            kernel_size=2,
            stride=2)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, image):
        # encoder
        x1 = self.block1(image)
        x2 = self.max_pool_2x2(x1)

        x3 = self.block2(x2)
        x4 = self.max_pool_2x2(x3)

        x5 = self.block3(x4)
        x6 = self.max_pool_2x2(x5)

        x7 = self.block4(x6)
        x8 = self.max_pool_2x2(x7)

        x9 = self.block5(x8)
        x10 = self.max_pool_2x2(x9)

        # нижняя часть
        x11 = self.block6(x10)

        # decoder
        x = self.up_trans_1(x11)
        x = torch.cat([x, x9], dim=1)
        x = self.block_up1(x)

        x = self.up_trans_2(x)
        x = torch.cat([x, x7], dim=1)
        x = self.block_up2(x)

        x = self.up_trans_3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.block_up3(x)

        x = self.up_trans_4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.block_up4(x)

        x = self.up_trans_5(x)
        x = torch.cat([x, x1], dim=1)
        x = self.block_up5(x)
        return x

        # print(x.size(),'мой вывод после "линии"')


if __name__ == "__main__":
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
    model = DLPUNet().to(device)
    input_tensor = torch.randn(1, 1, 256, 256).to(device)  # (B, C, H, W)

    params_trainable, params_total, flops, memory = calculate_model_metrics(model, input_tensor, device)
    print(f"{'Metric':<20} | {'Value':>15} | {'Unit':>10}")
    print("-" * 55)
    print(f"{'Trainable Params':<20} | {params_trainable:>15.2f} | {'M':>10}")
    print(f"{'Total Params':<20} | {params_total:>15.2f} | {'M':>10}")
    print(f"{'FLOPs':<20} | {flops:>15.4f} | {'GFLOPs':>10}")
    print(f"{'GPU Peak Memory':<20} | {memory:>15.2f} | {'MiB':>10}")
