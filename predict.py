import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

from model.cascadedUNet import CascadedUNet
from utils.dataset import BraTSDataset, get_case_list, split_dataset
from evaluate import evaluate
import json

def load_models(checkpoint_paths, device):
    """
    加载一个或多个模型权重，返回模型列表。
    checkpoint_paths: 单个路径字符串 或 路径列表
    """
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]

    models = []
    for path in checkpoint_paths:
        model = CascadedUNet().to(device)
        checkpoint = torch.load(path, map_location=device)

        # 兼容两种保存格式
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        models.append(model)

    print(f'加载了 {len(models)} 个模型权重')
    return models


def pred_to_seg(pred_bin):
    """
    三通道二值预测 -> 单通道标签（0/1/2/4）。
    pred_bin: [3, D, H, W]
    返回:     [D, H, W]，值为0/1/2/4
    """
    wt, tc, et = pred_bin[0], pred_bin[1], pred_bin[2]
    seg = np.zeros_like(wt, dtype=np.uint8)
    seg[wt == 1] = 2   # 水肿
    seg[tc == 1] = 1   # 坏死
    seg[et == 1] = 4   # 增强核心
    return seg



def predict(checkpoint_paths, case_list, output_folder,
            has_mask=True, use_tta=False, save_result=True,
            resume=True, progress_idx=0, siliding_window=False):
    """
    批量预测并保存结果。
    checkpoint_paths: 单个路径 或 路径列表（多个则做ensemble）
    case_list:        病例路径列表
    output_folder:    预测结果保存目录
    has_mask:         是否有标签，有的话顺便算Dice
    use_tta:          是否使用测试时翻转增强
    save_result: 是否保存.nii预测文件
    resume:      是否从上次中断的地方继续（断点续预测）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    os.makedirs(output_folder, exist_ok=True)
    # 加载模型
    models  = load_models(checkpoint_paths, device)
    # 用inference模式的Dataset，不裁剪，返回完整体积
    dataset = BraTSDataset(case_list, training=False, testing=True)

    # 加载已有的评估结果
    progress_file = os.path.join(output_folder, f'progress{progress_idx}.json')
    if resume and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            all_scores = json.load(f)
        finished_cases = {s['case_name'] for s in all_scores}
        print(f'发现已完成 {len(finished_cases)} 个病例，继续预测剩余病例')
    else:
        all_scores     = []
        finished_cases = set()

    for i in tqdm(range(len(dataset)), desc='预测'):
        image, mask, case_path = dataset[i]
        case_name = os.path.basename(case_path)
        # 跳过已经预测过的病例
        if case_name in finished_cases:
            continue
        # image: [4,240,240,155] tensor -> 加batch维度 -> [1,4,240,240,155]
        img_tensor = image.unsqueeze(0).to(device)
        case_preds = []
        # 每个模型分别预测，收集结果
        for model in models:
            if use_tta:
                preds_tta = []
                # 8种翻转组合
                flip_combinations = [
                    [False, False, False],
                    [True,  False, False],
                    [False, True,  False],
                    [False, False, True ],
                    [True,  True,  False],
                    [True,  False, True ],
                    [False, True,  True ],
                    [True,  True,  True ],
                ]
                for flips in flip_combinations:
                    img_flipped = img_tensor.clone()
                    # 翻转输入，空间轴是2/3/4（batch和channel是0/1）
                    for axis, do_flip in enumerate(flips):
                        if do_flip:
                            img_flipped = torch.flip(img_flipped, dims=[axis + 2])
                    if siliding_window:
                        pred = sliding_window_predict(model, img_flipped,
                                                      patch_size=96,
                                                      overlap=0.5,
                                                      device=device)
                    else:
                        with torch.no_grad():
                            pred = model(img_flipped)   # [1,3,D,H,W]
                    # 预测结果翻转回来
                    for axis, do_flip in enumerate(flips):
                        if do_flip:
                            pred = torch.flip(pred, dims=[axis + 2])
                    preds_tta.append(pred.squeeze(0).cpu().numpy())
                # 8种翻转取平均
                case_preds.append(np.mean(preds_tta, axis=0))
            else:
                if siliding_window:
                    pred = sliding_window_predict(model, img_tensor,
                                                  patch_size=96,
                                                  overlap=0.5,
                                                  device=device)
                else:
                    with torch.no_grad():
                        pred = model(img_tensor)
                case_preds.append(pred.squeeze(0).cpu().numpy())
        # 多个模型的预测取平均（ensemble）
        final_pred = np.mean(case_preds, axis=0)   # [3,D,H,W]
        # 概率>0.5转成二值
        pred_bin = (final_pred > 0.5).astype(np.uint8)
        # 后处理
        # pred_bin = post_process(pred_bin, et_threshold=200)

        # 评估
        if has_mask:
            mask_np = mask.numpy().astype(np.uint8)
            #print(f'pred_et:{pred_bin[2].sum()}, mask_et:{mask_np[2].sum()}')
            scores  = evaluate(pred_bin, mask_np)
            scores['case_name'] = case_name    # 记录病例名，断点续预测用
            all_scores.append(scores)
            print(f'\n{case_name} | '
                  f'WT={scores["dice_WT"]:.4f} '
                  f'TC={scores["dice_TC"]:.4f} '
                  f'ET={scores["dice_ET"]:.4f} '
                  f'mean={scores["dice_mean"]:.4f}')

            # 每预测完一个病例立刻保存进度
            with open(progress_file, 'w') as f:
                json.dump(all_scores, f)

        # 保存.nii文件
        if save_result:
            seg_out   = pred_to_seg(pred_bin)
            ref_nii   = nib.load(os.path.join(case_path, f'{case_name}_flair.nii'))
            out_nii   = nib.Nifti1Image(seg_out, ref_nii.affine, ref_nii.header)
            save_path = os.path.join(output_folder, f'{case_name}_pred.nii')
            nib.save(out_nii, save_path)

    # 汇总结果
    if has_mask and len(all_scores) > 0:
        print(f'\n===== 整体结果（共{len(all_scores)}例）=====')
        for key in ['dice_WT', 'dice_TC', 'dice_ET', 'dice_mean']:
            mean_val = np.mean([s[key] for s in all_scores])
            print(f'{key}: {mean_val:.4f}')


def post_process(pred_bin, et_threshold=200):
    """
    后处理：ET体积小于阈值时，把ET清零（归入TC）。

    pred_bin:     [3, D, H, W]，通道顺序 WT/TC/ET
    et_threshold: ET体素数量阈值，小于此值则清零
    """
    et = pred_bin[2]  # ET通道

    if et.sum() < et_threshold:
        pred_bin[2] = np.zeros_like(et)  # ET清零

    return pred_bin


def gaussian_kernel_3d(size=96, sigma=0.125):
    """
    生成3D高斯权重核，形状 [size, size, size]。
    中心权重最高，边缘权重最低。
    sigma越小，权重分布越集中在中心。
    """
    # 生成一维高斯分布，再扩展到三维
    coords = torch.arange(size).float() - size // 2
    # 归一化到[-1, 1]
    coords = coords / (size // 2)
    # 一维高斯
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    # 三维高斯 = 三个一维高斯的外积
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    # 归一化到[0, 1]
    g3d = g3d / g3d.max()
    return g3d  # [size, size, size]


def sliding_window_predict(model, image_tensor, patch_size=96,
                           overlap=0.5, device='cuda'):
    """
    滑动窗口推理。
    image_tensor: [1, 4, D, H, W]
    patch_size:   窗口大小，96
    overlap:      重叠比例，0.5表示步长=48
    返回: [3, D, H, W] numpy array，概率值
    """
    _, C, D, H, W = image_tensor.shape
    step = int(patch_size * (1 - overlap))   # 步长 = 96 * 0.5 = 48

    # 预计算高斯权重核
    kernel = gaussian_kernel_3d(patch_size).to(device)  # [96, 96, 96]

    # 累加预测结果和权重
    pred_accum   = torch.zeros(3, D, H, W, device=device)   # 加权预测之和
    weight_accum = torch.zeros(D, H, W, device=device)       # 权重之和

    # 生成所有窗口的起始坐标
    # 确保最后一个窗口能覆盖到边缘
    def get_starts(length, patch, step):
        starts = list(range(0, length - patch, step))
        # 最后一个窗口强制从 length-patch 开始，保证覆盖末尾
        if starts[-1] + patch < length:
            starts.append(length - patch)
        return starts

    z_starts = get_starts(D, patch_size, step)
    y_starts = get_starts(H, patch_size, step)
    x_starts = get_starts(W, patch_size, step)

    total_windows = len(z_starts) * len(y_starts) * len(x_starts)

    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    # 裁出当前窗口
                    patch = image_tensor[
                        :, :,
                        z:z + patch_size,
                        y:y + patch_size,
                        x:x + patch_size
                    ]  # [1, 4, 96, 96, 96]

                    # 预测
                    pred = model(patch)  # [1, 3, 96, 96, 96]
                    pred = pred.squeeze(0)  # [3, 96, 96, 96]

                    # 加权累加到对应位置
                    pred_accum[:, z:z + patch_size,
                    y:y + patch_size,
                    x:x + patch_size] += pred * kernel
                    weight_accum[z:z + patch_size,
                    y:y + patch_size,
                    x:x + patch_size] += kernel

    # 除以权重，得到加权平均预测
    # weight_accum需要扩展维度才能广播
    final = pred_accum / weight_accum.unsqueeze(0)  # [3, D, H, W]
    return final

if __name__ == '__main__':
    case_list = get_case_list('./data/MICCAI_BraTS_2019_Data_Training')
    train_cases, val_cases, test_cases = split_dataset(case_list,
                                                        val_split=0.1,
                                                        have_test=False)

    snapshot_paths = [
        f'checkpoints/snapshot_epoch{e}.pth'
        for e in range(405, 406)
    ]

    predict(
        checkpoint_paths = snapshot_paths,
        case_list = val_cases,
        output_folder = './predictions',
        has_mask = True,
        use_tta = False,
        save_result = False,
        resume = True,
        progress_idx = 2,
        siliding_window=True,
    )