# Two-Stage Cascaded U-Net for BraTS 2019

PyTorch implementation of "Two-Stage Cascaded U-Net: 
1st Place Solution to BraTS Challenge 2019 Segmentation Task"

## 项目结构
```
├── model/
│   ├── blocks.py          # PreActResBlock等基础模块
│   ├── stage1.py          # Stage1 U-Net
│   ├── stage2.py          # Stage2 双解码器网络
│   └── cascadedUNet.py    # 完整级联网络
├── utils/
│   ├── dataset.py         # 数据加载和预处理
│   └── dice_loss.py       # Soft Dice Loss
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
└── evaluate.py            # 评估指标
```

## 环境依赖
```bash
pip install torch nibabel scipy tqdm tensorboard
```

## 结果
```
result on validation set 
directly predict:
dice_WT: 0.8873
dice_TC: 0.8524
dice_ET: 0.7470
using sliding window:
dice_WT: 0.9018
dice_TC: 0.8700
dice_ET: 0.7529
```
