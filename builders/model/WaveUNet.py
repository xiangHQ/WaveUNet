__all__ = ["WaveUNet"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class conv_one(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_one, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the number of input channels does not match the output, we add a 1x1 convolution
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)

class GradientFusionBlock(nn.Module):
    def __init__(self, grad_channels):
        super(GradientFusionBlock, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2, grad_channels, 3, padding=1)
        )
    def forward(self, phase_input):
        grad_x = F.pad(phase_input[:, :, :, 1:] - phase_input[:, :, :, :-1], (0, 1),mode='constant')
        grad_y = F.pad(phase_input[:, :, 1:, :] - phase_input[:, :, :-1, :], (0, 0, 0, 1), mode='constant')
        grad_input = torch.cat([grad_x, grad_y], dim=1)  # (B, 2, H, W)
        edge_feat = self.edge_conv(grad_input)
        return edge_feat



class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpSampleConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor,
                              padding=scale_factor // 2, output_padding=scale_factor % 2),
        )

    def forward(self, x):
        return self.up(x)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        self.bottleneck = nn.Conv2d(in_channels + in_channels // len(pool_sizes) * len(pool_sizes),
                                    in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x] + [
            F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False)
            for stage in self.stages
        ]
        output = self.bottleneck(torch.cat(features, dim=1))
        output = self.bn(output)
        return self.relu(output)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.shape
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//8]
        key = self.key(x).view(batch, -1, H * W)  # [B, C//8, H*W]
        value = self.value(x).view(batch, -1, H * W)  # [B, C, H*W]

        attention = F.softmax(torch.bmm(query, key), dim=-1)  # [B, H*W, H*W]
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(batch, C, H, W)
        return self.gamma * out + x

class WaveUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.pre = nn.Sequential(conv_one(in_channels, base))
        self.grad = GradientFusionBlock(grad_channels=base)

        self.enc1 = ResidualBlock(4 * base, 2 * base)
        self.enc2 = ResidualBlock(8 * base, 4 * base)
        self.enc3 = ResidualBlock(16 * base, 8 * base)

        self.skip1 = nn.Conv2d(base, 2 * base, 3, stride=2, padding=1)
        self.skip2 = nn.Conv2d(2 * base, 4 * base, 3, stride=2, padding=1)
        self.skip3 = nn.Conv2d(4 * base, 8 * base, 3, stride=2, padding=1)

        self.dwt = DWTForward(J=1, wave='haar')

        self.ppm1 = PyramidPoolingModule(2 * base, (8, 16, 32, 64))
        self.ppm2 = PyramidPoolingModule(4 * base, (4, 8, 16, 32))
        self.ppm3 = PyramidPoolingModule(8 * base, (2, 4, 8, 16))

        self.attn = SelfAttention(8 * base)
        self.mid = conv_one(8 * base, 8 * base)

        self.up1 = UpSampleConv(16 * base, 4 * base)
        self.up2 = UpSampleConv(8 * base, 2 * base)
        self.up3 = UpSampleConv(4 * base, base)

        self.dec = ResidualBlock(16 * base, 16 * base)
        self.dec2 = ResidualBlock(8 * base, 8 * base)
        self.dec1 = ResidualBlock(4 * base, 4 * base)
        self.dec0 = ResidualBlock(2 * base, 2 * base)

        self.fuse2 = nn.Conv2d(4 * base, 4 * base, 3, padding=1)
        self.fuse1 = nn.Conv2d(2 * base, 2 * base, 3, padding=1)
        self.fuse0 = nn.Conv2d(base, base, 3, padding=1)
        self.output_head = nn.Sequential(
            ResidualBlock(2 * base, 2 * base),
            ResidualBlock(2 * base, 2 * base),
            nn.Conv2d(2 * base, out_channels, 1)
        )

    def _merge_dwt(self, yl, yh):
        return torch.cat([yl] + [yh[0][:, :, i, :, :] for i in range(3)], dim=1)

    def forward(self, x):
        x_grad = self.grad(x)
        x = self.pre(x)

        yl, yh = self.dwt(x)
        x_dwt = self._merge_dwt(yl, yh)
        res1 = self.skip1(x_grad)
        x1 = self.enc1(x_dwt) + res1
        x = self.ppm1(x1)

        yl, yh = self.dwt(x)
        x_dwt = self._merge_dwt(yl, yh)
        res2 = self.skip2(x)
        x2 = self.enc2(x_dwt) + res2
        x = self.ppm2(x2)

        yl, yh = self.dwt(x)
        x_dwt = self._merge_dwt(yl, yh)
        res3 = self.skip3(x)
        x3 = self.enc3(x_dwt) + res3
        x4 = self.ppm3(x3)

        x = self.attn(x4)
        x = self.mid(x)
        x = self.dec(torch.cat([x, x3], dim=1))

        x = self.up1(x)
        x = self.fuse2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up2(x)
        x = self.fuse1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.up3(x)
        x = self.fuse0(x)
        x = self.dec0(torch.cat([x, x_grad], dim=1))

        return self.output_head(x)

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
    model = WaveUNet().to(device)
    input_tensor = torch.randn(1, 1, 256, 256).to(device)  # (B, C, H, W)

    params_trainable, params_total, flops, memory = calculate_model_metrics(model, input_tensor, device)
    print(f"{'Metric':<20} | {'Value':>15} | {'Unit':>10}")
    print("-" * 55)
    print(f"{'Trainable Params':<20} | {params_trainable:>15.2f} | {'M':>10}")
    print(f"{'Total Params':<20} | {params_total:>15.2f} | {'M':>10}")
    print(f"{'FLOPs':<20} | {flops:>15.4f} | {'GFLOPs':>10}")
    print(f"{'GPU Peak Memory':<20} | {memory:>15.2f} | {'MiB':>10}")
    # from ptflops import get_model_complexity_info
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = WaveUNet().to(device)
    # flops, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)




