import torch
import torch.nn as nn
from .blocks import PreActResBlock, EncoderBlock, DownSample, DecoderUp


class Stage2UNet(nn.Module):
    """
    Stage 2 U-Net，对应论文 Table 2。

    输入通道 = 4（原始模态）+ 3（Stage1粗糙输出）= 7
    base_filters = 32（是Stage1的两倍，论文"double the number of filters"）

    encoder(与stage1相比，width翻倍)：
      InitConv:  7 x 128^3 -> 32 x 128^3
      EnBlock1:  32 x 128^3, preactResBlock x 1, get o1
      EnDown1:   32 x 128^3 -> 64 x 64^3, downSample x 1
      EnBlock2:  64 x 64^3, preactResBlock x 2, get o2
      EnDown2:   64 x 64^3 -> 128 x 32^3, downSample x 1
      EnBlock3:  128 x 32^3, preactResBlock x 2, get o3
      EnDown3:   128 x 32^3 -> 256 x 16^3, downSample x 1
      EnBlock4:  256 x 16^3, preactResBlock x 4

    Decoder1（转置卷积，训练+推理都用）：
      和Stage1解码器结构完全一致，只是通道数翻倍

    Decoder2（三线性插值，only training，regularize shared encoder）：
      和Decoder1结构完全一致，只是上采样方式不同
    """

    def __init__(self, in_channels=7, base_filters=32, out_channels=3,
                 num_groups=8, dropout=0.2):
        super().__init__()

        f = base_filters  # 32

        # shared encoder
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(p=dropout),
        )

        self.en_block1 = EncoderBlock(f,   num_blocks=1, num_groups=num_groups)
        self.en_down1  = DownSample(f,     num_groups=num_groups)   # -> f*2

        self.en_block2 = EncoderBlock(f*2, num_blocks=2, num_groups=num_groups)
        self.en_down2  = DownSample(f*2,   num_groups=num_groups)   # -> f*4

        self.en_block3 = EncoderBlock(f*4, num_blocks=2, num_groups=num_groups)
        self.en_down3  = DownSample(f*4,   num_groups=num_groups)   # -> f*8

        self.en_block4 = EncoderBlock(f*8, num_blocks=4, num_groups=num_groups)

        # decoder1:transpose
        self.de1_up3    = DecoderUp(f*8, use_transpose=True, num_groups=num_groups)
        self.de1_block3 = EncoderBlock(f*4, num_blocks=1, num_groups=num_groups)

        self.de1_up2    = DecoderUp(f*4, use_transpose=True, num_groups=num_groups)
        self.de1_block2 = EncoderBlock(f*2, num_blocks=1, num_groups=num_groups)

        self.de1_up1    = DecoderUp(f*2, use_transpose=True, num_groups=num_groups)
        self.de1_block1 = EncoderBlock(f,   num_blocks=1, num_groups=num_groups)

        self.de1_end_conv = nn.Conv3d(f, out_channels, kernel_size=1, bias=False)
        self.de1_sigmoid  = nn.Sigmoid()

        # decoder2:trilinear
        self.de2_up3    = DecoderUp(f*8, use_transpose=False, num_groups=num_groups)
        self.de2_block3 = EncoderBlock(f*4, num_blocks=1, num_groups=num_groups)

        self.de2_up2    = DecoderUp(f*4, use_transpose=False, num_groups=num_groups)
        self.de2_block2 = EncoderBlock(f*2, num_blocks=1, num_groups=num_groups)

        self.de2_up1    = DecoderUp(f*2, use_transpose=False, num_groups=num_groups)
        self.de2_block1 = EncoderBlock(f,   num_blocks=1, num_groups=num_groups)

        self.de2_end_conv = nn.Conv3d(f, out_channels, kernel_size=1, bias=False)
        self.de2_sigmoid  = nn.Sigmoid()

    def forward(self, x, training=True):
        """
        x:        [B, 7, 128, 128, 128]  (原始4通道 + Stage1粗糙输出3通道)
        training: True时同时运行Decoder2，False时只运行Decoder1
        返回:
          out_deconv:  转置卷积解码器输出，[B, 3, 128, 128, 128]
          out_interp:  三线性插值解码器输出（仅training=True时），[B, 3, 128, 128, 128] or None
        """

        # shared encoder
        x0 = self.init_conv(x)         # [B, 32, 128, 128, 128]
        e1 = self.en_block1(x0)        # [B, 32, 128, 128, 128]

        d1 = self.en_down1(e1)         # [B, 64,  64,  64,  64]
        e2 = self.en_block2(d1)        # [B, 64,  64,  64,  64]

        d2 = self.en_down2(e2)         # [B,128,  32,  32,  32]
        e3 = self.en_block3(d2)        # [B,128,  32,  32,  32]

        d3 = self.en_down3(e3)         # [B,256,  16,  16,  16]
        e4 = self.en_block4(d3)        # [B,256,  16,  16,  16]

        # decoder1
        u3 = self.de1_up3(e4, e3)      # [B,128,  32,  32,  32]
        u3 = self.de1_block3(u3)
        u2 = self.de1_up2(u3, e2)      # [B, 64,  64,  64,  64]
        u2 = self.de1_block2(u2)
        u1 = self.de1_up1(u2, e1)      # [B, 32, 128, 128, 128]
        u1 = self.de1_block1(u1)
        out_deconv = self.de1_sigmoid(self.de1_end_conv(u1))

        # decoder2
        out_interp = None
        if training:
            v3 = self.de2_up3(e4, e3)  # [B,128,  32,  32,  32]
            v3 = self.de2_block3(v3)
            v2 = self.de2_up2(v3, e2)  # [B, 64,  64,  64,  64]
            v2 = self.de2_block2(v2)
            v1 = self.de2_up1(v2, e1)  # [B, 32, 128, 128, 128]
            v1 = self.de2_block1(v1)
            out_interp = self.de2_sigmoid(self.de2_end_conv(v1))

        return out_deconv, out_interp