"""
Soft Dice Loss(单通道)：
    L = 1 - (2 * sum(s * r)) / (sum(s^2) + sum(r^2) + eps)
对三个通道（WT/TC/ET）分别计算loss后求和：
    Loss_total = L(WT) + L(TC) + L(ET)
"""
def soft_dice_loss(pred, mask, eps=1e-6):
    """
    pred:   [B, 3, D, H, W]  sigmoid后的预测概率
    mask:   [B, 3, D, H, W]  三通道二值mask（WT/TC/ET）
    返回标量loss
    """
    # 把空间维度全部展平
    # [B, 3, D, H, W] -> [B, 3, N]，N = D*H*W
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    mask = mask.view(mask.shape[0], mask.shape[1], -1).float()

    # 分子：2 * sum(s * r)
    # sum(dim=2): 在像素维度(N)上求和
    # [B, 3, N] -> [B, 3]
    numerator = 2.0 * (pred * mask).sum(dim=2)

    # 分母：sum(s^2) + sum(r^2)
    # [B, 3, N] -> [B, 3]
    denominator = (pred ** 2).sum(dim=2) + (mask ** 2).sum(dim=2)

    # 各通道的soft dice得分，形状 [B, 3]
    soft_dice = numerator / (denominator + eps)

    # 每个通道变成loss（1 - dice），三个通道相加，对batch取均值
    loss = (1 - soft_dice).sum(dim=1).mean()

    return loss


def compute_loss(model_output, label, alpha=1.0):
    """
    model_output: (coarse_out, out_deconv, out_interp)
    label:        [B, 3, D, H, W]  三通道二值mask（WT/TC/ET）
    alpha:        Decoder2 loss的权重

    返回:
      loss_total:  tensor标量，用于反向传播
      loss_dict:   各分项的loss值，用于打印监控
    """
    coarse_out, out_deconv, out_interp = model_output

    loss1 = soft_dice_loss(coarse_out, label)   # Stage1 loss
    loss2 = soft_dice_loss(out_deconv, label)   # Stage2 Decoder1 loss
    loss3 = soft_dice_loss(out_interp, label)   # Stage2 Decoder2 loss（正则项）

    loss_total = loss1 + loss2 + alpha * loss3

    # .item() 把tensor转成普通Python float，用于监控变化
    loss_dict = {
        'loss_total':   loss_total.item(),
        'loss1_coarse': loss1.item(),
        'loss2_deconv': loss2.item(),
        'loss3_interp': loss3.item(),
    }

    return loss_total, loss_dict