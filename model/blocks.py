import torch.nn as nn
import torch.nn.functional as F


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block.
    结构: GN -> ReLU -> Conv3d -> GN -> ReLU -> Conv3d -> + shortcut
    """

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)

        # 通道数不一致时，shortcut需要投影对齐
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return out + residual # 残差拼接


class EncoderBlock(nn.Module):
    """
    Encoder & Decoder 共用
    N个PreActResBlock堆叠，不改变空间分辨率和通道数。
    """

    def __init__(self, channels, num_blocks, num_groups=8):
        super().__init__()
        self.blocks = nn.Sequential(
            *[PreActResBlock(channels, channels, num_groups)
              for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class DownSample(nn.Module):
    """
    使用 Conv3x3x3 stride=2 下采样，同时通道数 x2。
    """

    def __init__(self, in_channels, num_groups=8):
        super().__init__()
        out_channels = in_channels * 2
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class DecoderUp(nn.Module):
    """
    上采样模块：Conv1x1x1 减半通道 + 上采样（转置卷积 或 三线性插值）。
    然后加上编码器对应层的skip connection（elementwise sum）。
    """

    def __init__(self, in_channels, use_transpose=True, num_groups=8):
        super().__init__()
        out_channels = in_channels // 2

        # 1x1x1 先减半通道
        self.conv1x1 = nn.Conv3d(in_channels, out_channels,
                                 kernel_size=1, bias=False)

        self.use_transpose = use_transpose
        if use_transpose:
            # kernel_size=2x2x2, stride=2 的转置卷积
            self.upsample = nn.ConvTranspose3d(out_channels, out_channels,
                                               kernel_size=2, stride=2, bias=False)
        # 三线性插值不需要额外参数，forward里直接用F.interpolate

    def forward(self, x, skip):
        """
        x:    解码器上一层
        skip: 来自编码器对应层的特征图
        """
        x = self.conv1x1(x)
        if self.use_transpose:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode='trilinear',
                              align_corners=False)

        # x可能比skip大1个像素（输入含奇数尺寸时发生）
        # 直接把x裁剪到和skip完全一样的尺寸
        # 下采样时卷积公式：
        # output = floor((input + 2*padding - kernel) / stride) + 1
        #        = floor((input + 2*1 - 3) / 2) + 1
        #        = floor((input - 1) / 2) + 1
        # D轴: 240 → floor(239/2)+1=120 → floor(119/2)+1=60 → floor(59/2)+1=30
        # H轴: 240 → 120 → 60 → 30     # 和D轴一样
        # W轴: 155 → floor(154/2)+1=78 → floor(77/2)+1=39 → floor(38/2)+1=20
        # 转置卷积上采样时：
        # 60 × 60 × 40  ← x，skip3是39，x比skip大1，裁去1
        # 120 × 120 × 78   ← x，skip2是78，匹配
        # 240 × 240 × 156  ← x，skip1是155，x比skip大1，裁去1
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3], :skip.shape[4]]
        # elementwise sum（加法而非拼接）
        return x + skip
