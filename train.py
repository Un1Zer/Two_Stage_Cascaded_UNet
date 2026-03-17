import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import DataLoader
from utils.dataset import get_case_list, split_dataset, BraTSDataset
from model.cascadedUNet import CascadedUNet
from utils.dice_loss import compute_loss
from evaluate import validate

def train(resume=None):
    # 1. 基本设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 保存模型权重的文件夹
    os.makedirs('checkpoints', exist_ok=True)

    # 2. 数据准备
    case_list = get_case_list('./data/MICCAI_BraTS_2019_Data_Training')
    train_cases, val_cases, test_cases = split_dataset(case_list, val_split=0.1,have_test=False)
    print(f'训练集: {len(train_cases)}例，验证集: {len(val_cases)}例')

    train_dataset = BraTSDataset(train_cases, training=True,patch_size=(96, 96, 96))
    val_dataset = BraTSDataset(val_cases, training=False,validation=True,patch_size=(96, 96, 96))

    # DataLoader负责把dataset里的样本自动打包成batch
    # pin_memory=True 让数据加载到固定内存，加快CPU到GPU的传输
    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4, pin_memory=True)

    # 3. 获取模型
    model = CascadedUNet().to(device)

    # 4. 优化器
    # Adam优化器，根据原论文指定lr=1e-4，weight_decay=1e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler()

    # 5. 训练循环
    max_epochs = 405
    start_epoch = 1

    # 恢复训练
    if resume is not None:
        if not os.path.exists(resume):
            print(f'警告：找不到checkpoint文件 {resume}，从头开始训练')
        else:
            checkpoint = torch.load(resume, map_location=device)

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1   # 从下一个epoch开始

            print(f'从checkpoint恢复：epoch {checkpoint["epoch"]}，'
                  f'val_loss={checkpoint["val_loss"]:.4f}，'
                  f'继续从epoch {start_epoch}开始训练')

    # 创建writer，logs文件夹下按时间戳区分不同训练runs
    log_dir = 'logs/train'
    writer  = SummaryWriter(log_dir)
    print(f'TensorBoard日志保存到: {log_dir}')

    for epoch in range(start_epoch, max_epochs + 1):
        # 训练一个epoch
        train_loss_total, train_loss2\
            = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)

        print(f'Epoch {epoch}/{max_epochs} | '
              f'train_loss_total={train_loss_total:.4f} | '
              f'train_loss2={train_loss2:.4f} | ')

        # 每个epoch都记录训练loss
        writer.add_scalar('Loss/train_total', train_loss_total, epoch)
        writer.add_scalar('Loss/train_loss2', train_loss2, epoch)

        # 每5个epoch验证一次
        if epoch % 5 == 0:
            val_loss = validate(model, val_loader, device)
            print(f'Epoch {epoch}/{max_epochs} | '
                f'val_loss={val_loss:.4f}')
            writer.add_scalar('Loss/val_loss', val_loss, epoch)
            # 每次验证后保存checkpoint，包含恢复训练所需的所有信息
            checkpoint = {
                'epoch'     : epoch,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler'    : scaler.state_dict(),
                'val_loss'  : val_loss,
            }
            torch.save(checkpoint,
                       f'checkpoints/checkpoint_epoch{epoch}.pth')
            print(f'checkpoint_epoch{epoch} saved!')
            writer.flush()

        # 保存最后8个epoch的权重（用于推理集成）
        if epoch > max_epochs - 8:
            path = f'checkpoints/snapshot_epoch{epoch}.pth'
            torch.save(model.state_dict(), path)
            print(f'已保存snapshot: {path}')

    writer.close()


def get_lr(epoch, max_epochs, lr_init=1e-4, warmup_epochs=5):
    """
    论文:The maximum number of training iterations is set to
    405 epochs with 5 epochs of linear warmup

    warmup阶段（前5个epoch）：学习率从0线性增加到lr_init
    为什么warmup：训练刚开始模型权重是随机的，梯度很不稳定，
    如果一开始就用大学习率很容易把训练搞崩，所以先用小学习率热身。

    warmup结束后：按论文公式衰减
    lr = lr_init * (1 - epoch/max_epochs) ^ 0.9
    随着epoch增加学习率慢慢降低
    """
    if epoch < warmup_epochs:
        # 线性warmup：epoch=0时lr=0，epoch=4时lr接近lr_init
        return lr_init * (epoch + 1) / warmup_epochs
    else:
        progress = epoch / max_epochs
        return lr_init * (1 - progress) ** 0.9


def set_lr(optimizer, lr):
    """把optimizer里所有参数组的学习率都改成lr"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, max_epochs=405):
    """
    训练一个完整的epoch，遍历loader里的所有batch。
    返回这个epoch的平均loss。
    """
    # 切换到训练模式
    # 训练模式下Dropout会随机丢弃神经元
    model.train()
    print(f'training epoch : {epoch}/{max_epochs}')

    # 更新学习率
    lr = get_lr(epoch - 1, max_epochs)  # 公式希望epoch counter从0开始
    set_lr(optimizer, lr)

    total_loss = 0.0
    loss2 = 0.0
    n_batches = 0
    # 用tqdm包裹loader，自动显示进度条
    # leave=True表示epoch结束后进度条保留在屏幕上不消失
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{max_epochs} [训练]', leave=True)

    for image, mask in pbar:
        # 把数据移到GPU
        image = image.to(device)  # [1, 4, 128, 128, 128]
        mask = mask.to(device)  # [1, 3, 128, 128, 128]

        # 用autocast包裹前向传播
        # autocast自动决定哪些操作用float16，哪些保持float32
        with torch.amp.autocast(device.type):
            model_output = model(image)
            loss, loss_dict = compute_loss(model_output, mask, alpha=1.0)

        optimizer.zero_grad()

        # 用scaler.scale(loss).backward()代替loss.backward()
        scaler.scale(loss).backward()

        # 用scaler.step()和scaler.update()代替optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict['loss_total']
        loss2 += loss_dict['loss2_deconv']
        n_batches += 1

        # 实时更新进度条右边显示的指标
        # postfix接受一个dict，会显示成 key=value 的格式
        pbar.set_postfix({
            'total_loss': f'{loss_dict["loss_total"]:.4f}',
            'L1': f'{loss_dict["loss1_coarse"]:.4f}',
            'L2': f'{loss_dict["loss2_deconv"]:.4f}',
            'L3': f'{loss_dict["loss3_interp"]:.4f}',
            'lr': f'{lr:.6f}',
        })

    # 返回这个epoch里total loss和loss2的均值
    return total_loss / n_batches, loss2 / n_batches

if __name__ == '__main__':
    train('checkpoints/checkpoint_epoch335.pth')
