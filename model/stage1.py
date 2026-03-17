import torch
import torch.nn as nn
from .blocks import PreActResBlock, EncoderBlock, DownSample, DecoderUp


class Stage1UNet(nn.Module):
    """
    Stage 1 U-Net，对应论文 Table 1。

    encoder：
      InitConv:  4 x 128^3 -> 16 x 128^3
      EnBlock1:  16 x 128^3, preactResBlock x 1, get o1
      EnDown1:   16 x 128^3 -> 32 x 64^3, downSample x 1
      EnBlock2:  32 x 64^3, preactResBlock x 2, get o2
      EnDown2:   32 x 64^3 -> 64 x 32^3, downSample x 1
      EnBlock3:  64 x 32^3, preactResBlock x 2, get o3
      EnDown3:   64 x 32^3 -> 128 x 16^3, downSample x 1
      EnBlock4:  128 x 16^3, preactResBlock x 4

    decoder：
      DeUp3:     128 x 16^3 -> 64 x 32^3, upSample + o3
      DeBlock3:  64 x 32^3, preactResBlock x 1
      DeUp2:     64 x 32^3 -> 32 x 64^3, upSample + o2
      DeBlock2:  32 x 64^3, preactResBlock x 1
      DeUp1:     32 x 64^3 -> 16 x 128^3, upSample + o1
      DeBlock1:  16 x 128^3, preactResBlock x 1
      EndConv:   16 x 128^3 -> 3 x 128^3, 1*1conv
      Sigmoid
    """

    def __init__(self, in_channels=4, base_filters=16, out_channels=3,
                 num_groups=8, dropout=0.2):
        super().__init__()

        f = base_filters  # 16

        # encoder
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(p=dropout),
        )

        self.en_block1 = EncoderBlock(f, num_blocks=1, num_groups=num_groups)
        self.en_down1  = DownSample(f, num_groups=num_groups)   # -> f*2

        self.en_block2 = EncoderBlock(f*2, num_blocks=2, num_groups=num_groups)
        self.en_down2  = DownSample(f*2, num_groups=num_groups)   # -> f*4

        self.en_block3 = EncoderBlock(f*4, num_blocks=2, num_groups=num_groups)
        self.en_down3  = DownSample(f*4, num_groups=num_groups)   # -> f*8

        self.en_block4 = EncoderBlock(f*8, num_blocks=4, num_groups=num_groups)

        # decoder
        # DeUp3: f*8 -> f*4, 加 EnBlock3 的 skip
        self.de_up3    = DecoderUp(f*8, use_transpose=True, num_groups=num_groups)
        self.de_block3 = EncoderBlock(f*4, num_blocks=1, num_groups=num_groups)

        # DeUp2: f*4 -> f*2, 加 EnBlock2 的 skip
        self.de_up2    = DecoderUp(f*4, use_transpose=True, num_groups=num_groups)
        self.de_block2 = EncoderBlock(f*2, num_blocks=1, num_groups=num_groups)

        # DeUp1: f*2 -> f, 加 EnBlock1 的 skip
        self.de_up1    = DecoderUp(f*2, use_transpose=True, num_groups=num_groups)
        self.de_block1 = EncoderBlock(f, num_blocks=1, num_groups=num_groups)

        # out
        self.end_conv = nn.Conv3d(f, out_channels, kernel_size=1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # 编码
        x0  = self.init_conv(x)        # [B, 16, 128, 128, 128]
        e1  = self.en_block1(x0)       # [B, 16, 128, 128, 128]

        d1  = self.en_down1(e1)        # [B, 32,  64,  64,  64]
        e2  = self.en_block2(d1)       # [B, 32,  64,  64,  64]

        d2  = self.en_down2(e2)        # [B, 64,  32,  32,  32]
        e3  = self.en_block3(d2)       # [B, 64,  32,  32,  32]

        d3  = self.en_down3(e3)        # [B,128,  16,  16,  16]
        e4  = self.en_block4(d3)       # [B,128,  16,  16,  16]

        # 解码（每步都加对应编码器的skip connection）
        u3  = self.de_up3(e4, e3)      # [B, 64,  32,  32,  32]
        u3  = self.de_block3(u3)       # [B, 64,  32,  32,  32]

        u2  = self.de_up2(u3, e2)      # [B, 32,  64,  64,  64]
        u2  = self.de_block2(u2)       # [B, 32,  64,  64,  64]

        u1  = self.de_up1(u2, e1)      # [B, 16, 128, 128, 128]
        u1  = self.de_block1(u1)       # [B, 16, 128, 128, 128]

        # 输出
        out = self.end_conv(u1)        # [B,  3, 128, 128, 128]
        out = self.sigmoid(out)        # 三通道sigmoid，对应WT/TC/ET

        return out