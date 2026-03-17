import numpy as np
import torch
from tqdm import tqdm

from utils.dice_loss import soft_dice_loss


# 评估指标
def dice_score(pred_bin, mask_bin):
    """
    计算单个通道的Dice Score。
    pred_bin, mask_bin: numpy array，值为0或1。

    当两者都为空（全0）时返回1.0，认为预测正确。
    """
    intersection = (pred_bin * mask_bin).sum()
    denom = pred_bin.sum() + mask_bin.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom

# 这个装饰器让整个函数里的操作都不计算梯度
# 等价于把整个函数体放在 with torch.no_grad(): 里
# 验证时不需要梯度，加上这个可以节省显存和加快速度
@torch.no_grad()
def validate(model, loader, device):
    """
    在验证集上跑一遍，返回平均loss和dice score
    """
    model.eval()  # 切换到推理模式，关闭Dropout

    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f'[验证]', leave=True)

    for image, mask in pbar:
        image = image.to(device)
        mask = mask.to(device)

        # 验证时也用autocast，和训练时保持一致
        with torch.amp.autocast(device.type):
            pred = model(image) # [1, 3, 240, 240, 155]
            loss = soft_dice_loss(pred, mask)


        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({
            'val_loss': f'{total_loss / n_batches:.4f}'
        })

    model.train()  # 验证完切回训练模式
    return total_loss / n_batches

@torch.no_grad()
def evaluate(pred_bin, mask):
    """
    计算三个通道（WT/TC/ET）的Dice Score。
    pred_bin: [3, D, H, W]，值为0或1
    mask:     [3, D, H, W]，值为0或1

    返回dict，包含三个通道的Dice和平均Dice。
    """
    names = ['WT', 'TC', 'ET']
    scores = {}
    for i, name in enumerate(names):
        scores[f'dice_{name}'] = dice_score(pred_bin[i], mask[i])
    scores['dice_mean'] = np.mean(list(scores.values()))
    return scores