import torch
import torch.nn as nn
from .stage1 import Stage1UNet
from .stage2 import Stage2UNet


class CascadedUNet(nn.Module):
    """
    Two-Stage Cascaded U-Net（论文完整实现）。

    前向流程：
      1. Stage1(x)                    → coarse_out       [B,3,128,128,128]
      2. concat(x, coarse_out)        → x2               [B,7,128,128,128]
      3. Stage2(x2, training)         → out_deconv, out_interp

    三个输出对应三个loss：
      Loss1 = SoftDice(coarse_out,  label)
      Loss2 = SoftDice(out_deconv,  label)
      Loss3 = SoftDice(out_interp,  label)

    Loss_total = Loss1 + Loss2 + alpha * Loss3
    """

    def __init__(self, in_channels=4, out_channels=3,
                 s1_base_filters=16, s2_base_filters=32,
                 num_groups=8, dropout=0.2, alpha=1.0):
        """
        alpha: Decoder2（插值）loss的权重，论文未明确给出，默认1.0
        """
        super().__init__()

        self.alpha = alpha

        self.stage1 = Stage1UNet(
            in_channels=in_channels,
            base_filters=s1_base_filters,
            out_channels=out_channels,
            num_groups=num_groups,
            dropout=dropout,
        )

        # Stage2输入 = 原始4通道 + Stage1粗糙输出3通道 = 7通道
        self.stage2 = Stage2UNet(
            in_channels=in_channels + out_channels,  # 7
            base_filters=s2_base_filters,
            out_channels=out_channels,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, x):
        """
        x: [B, 4, 128, 128, 128]  四种MRI模态

        训练时返回三个输出：(coarse_out, out_deconv, out_interp)
        推理时返回一个输出：out_deconv（最终精细预测）
        """
        training = self.training

        # Stage1：粗糙预测
        coarse_out = self.stage1(x)                  # [B, 3, 128, 128, 128]

        # 拼接原始输入和粗糙输出，传入Stage2（auto-context）
        x2 = torch.cat([x, coarse_out], dim=1)       # [B, 7, 128, 128, 128]

        # Stage2：精细预测（训练时同时运行两个解码器）
        out_deconv, out_interp = self.stage2(x2, training=training)

        if training:
            return coarse_out, out_deconv, out_interp
        else:
            # 推理时Decoder2不参与，直接返回Decoder1的结果
            return out_deconv