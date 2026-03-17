import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class BraTSDataset(Dataset):
    """
    继承Dataset类须实现两个方法：
      __len__：返回数据集大小
      __getitem__：给定索引idx，返回第idx个样本
    DataLoader会自动调用这两个方法来批量加载数据。
    """
    def __init__(self, case_list, patch_size=(128, 128, 128), training=True, validation = False, testing = False):
        """
        case_list:  病例文件夹路径列表
        patch_size: 裁剪大小
        training:   True时做随机裁剪+数据增强，False返回原始图像
        """
        self.case_list = case_list
        self.patch_size = patch_size
        self.training = training
        self.validation = validation
        self.testing = testing

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case_path = self.case_list[idx]
        # 1. 加载
        image, seg = load_case(case_path)  # [4,240,240,155], [240,240,155]
        # 2. 归一化
        image = normalize(image)  # [4,240,240,155]
        # 3. 标签转换
        mask = seg_to_mask(seg)  # [3,240,240,155]
        # 4. 裁剪（仅训练/验证）
        if self.training:
            image, mask = random_crop(image, mask, self.patch_size)
        elif self.validation:
            image, mask = center_crop(image, mask, self.patch_size)
        # 5. 数据增强（仅训练）
        if self.training:
            image, mask = augment(image, mask)
        # 6. 转成PyTorch tensor
        # np.flip之后内存不连续，需要先.copy()才能转tensor
        image = torch.from_numpy(image.copy()).float()  # [4, 128, 128, 128]
        mask = torch.from_numpy(mask.copy()).float()  # [3, 128, 128, 128]

        if self.testing:
            return image, mask, case_path
        return image, mask

def get_case_list(data_root):
    """
    扫描HGG和LGG两个文件夹，返回所有病例的完整路径列表
    """
    cases = []
    for grade in ['HGG', 'LGG']:
        grade_dir = os.path.join(data_root, grade)
        if not os.path.exists(grade_dir):
            continue
        # 排序保证取出的顺序一致，与seed配合使实验可复现
        for case_name in sorted(os.listdir(grade_dir)):
            case_path = os.path.join(grade_dir, case_name)
            # 将该病例路径加入cases列表
            if os.path.isdir(case_path):
                cases.append(case_path)
    return cases


def split_dataset(case_list, val_split=0.1, seed=42, have_test=True, test_split=0.1):
    """
    随机划分训练集, 验证集和测试集
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(case_list))
    n = len(case_list)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    if have_test:
        test_cases = [case_list[i] for i in indices[:n_test]]
        val_cases = [case_list[i] for i in indices[n_test:n_test + n_val]]
        train_cases = [case_list[i] for i in indices[n_test + n_val:]]
    else:
        test_cases = []
        val_cases   = [case_list[i] for i in indices[:n_val]]
        train_cases = [case_list[i] for i in indices[n_val:]]

    return train_cases, val_cases, test_cases


def load_case(case_path):
    """
    加载单个病例的四种模态和分割标签，并将四个模态拼接

    case_path: 病例文件夹的完整路径，例如：
    /data/BraTS2019/HGG/BraTS19_2013_10_1

    返回：
      image: np.array [4, D, H, W]，float32，四种模态拼在通道维
      seg:   np.array [D, H, W]，值为 0/1/2/4
    """
    case_name = os.path.basename(case_path)  # 取文件夹名，如 BraTS19_2013_10_1

    modalities = []
    for mod in ['flair', 't1', 't1ce', 't2']:
        file_path = os.path.join(case_path, f'{case_name}_{mod}.nii')
        vol = nib.load(file_path).get_fdata(dtype=np.float32)  # [D, H, W]
        modalities.append(vol)

    image = np.stack(modalities, axis=0)  # [4, D, H, W]

    seg_path = os.path.join(case_path, f'{case_name}_seg.nii')
    seg = nib.load(seg_path).get_fdata(dtype=np.float32)  # [D, H, W]

    return image, seg


def normalize(image):
    """
    对每个模态的脑区做归一化，使模型接收到的输入更稳定
    颅骨已经被去除，背景全是0
    只用非零体素（脑区）计算均值和标准差

    image: [4, D, H, W]
    """
    for c in range(4):
        vol = image[c]  # 取出单个模态，[D, H, W]
        brain_mask = vol > 0  # 非零体素就是脑区，[D, H, W]，bool类型

        mean = vol[brain_mask].mean()
        std = vol[brain_mask].std()
        std = std if std > 1e-8 else 1e-8

        image[c] = (vol - mean) / std

    return image


def seg_to_mask(seg):
    """
    将单通道标签（0/1/2/4）转为三通道二值mask。
    三个区域：WT ⊃ TC ⊃ ET
      通道0 WT（全肿瘤）    = 标签 1 + 2 + 4
      通道1 TC（肿瘤核心）  = 标签 1 + 4
      通道2 ET（增强核心）  = 标签 4

    seg: [D, H, W]，值为0/1/2/4
    返回: [3, D, H, W]，每个通道标签值为0或1
    """
    wt = (seg > 0).astype(np.float32)  # 1+2+4
    tc = ((seg == 1) | (seg == 4)).astype(np.float32)  # 1+4
    et = (seg == 4).astype(np.float32)  # 4

    return np.stack([wt, tc, et], axis=0)  # [3, D, H, W]


def random_crop(image, mask, patch_size=(128, 128, 128)):
    """
    前景偏置随机裁剪：
    50%以肿瘤体素为中心裁剪（保证patch里有肿瘤）, 50%完全随机裁剪
    [c, d, h, w] -> [c, pd, ph, pw]
    """
    c, d, h, w = image.shape
    pd, ph, pw = patch_size

    if np.random.rand() < 0.5:
        # 找到所有肿瘤体素的坐标（用WT通道，最大的那个区域）
        fg_coords = np.argwhere(mask[0] > 0)  # [[z1,y1,x1], [z2,y2,x2], ...]

        # 随机选一个肿瘤体素作为参考点
        ref = fg_coords[np.random.randint(len(fg_coords))]  # [z, y, x]

        # 以ref为中心计算起始坐标，再用clip保证不越界
        z = int(np.clip(ref[0] - pd // 2, 0, d - pd))
        y = int(np.clip(ref[1] - ph // 2, 0, h - ph))
        x = int(np.clip(ref[2] - pw // 2, 0, w - pw))
    else:
        z = np.random.randint(0, d - pd + 1)
        y = np.random.randint(0, h - ph + 1)
        x = np.random.randint(0, w - pw + 1)

    image = image[:, z:z+pd, y:y+ph, x:x+pw]
    mask  = mask[:,  z:z+pd, y:y+ph, x:x+pw]

    return image, mask

def center_crop(image, mask, patch_size):
    """
    验证时做中心裁剪，保证每次裁到的位置一样，结果可复现。
    """
    _, d, h, w = image.shape
    pd, ph, pw = patch_size

    z = (d - pd) // 2  # 240//2 - 128//2 = 56
    y = (h - ph) // 2  # 56
    x = (w - pw) // 2  # 155//2 - 128//2 = 13

    image = image[:, z:z + pd, y:y + ph, x:x + pw]
    mask = mask[:, z:z + pd, y:y + ph, x:x + pw]

    return image, mask


def augment(image, mask):
    """
    三种数据增强
    1. 随机强度偏移：每个模态加上 [-0.1, 0.1] * 该模态标准差的随机值
    2. 随机强度缩放：每个模态乘以 [0.9, 1.1] 之间的随机值
    3. 随机翻转：沿x/y/z三个轴各自50%概率翻转(image和mask同时进行)

    image: [4, D, H, W]
    mask:  [3, D, H, W]
    """
    # 强度偏移 + 缩放（只对image）
    for c in range(4):
        std = image[c].std()
        shift = np.random.uniform(-0.1, 0.1) * std  # 随机偏移量
        scale = np.random.uniform(0.9, 1.1)  # 随机缩放比例
        image[c] = image[c] * scale + shift

    # 随机翻转（image和mask同步）
    # image的轴：[4, D, H, W]，空间轴是1/2/3
    # mask的轴： [3, D, H, W]，空间轴是1/2/3
    for axis in [1, 2, 3]:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)

    return image, mask