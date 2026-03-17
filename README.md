# Two-Stage Cascaded U-Net for BraTS 2019

PyTorch implementation of "Two-Stage Cascaded U-Net: 
1st Place Solution to BraTS Challenge 2019 Segmentation Task"

## Structure
```
├── model/
│   ├── blocks.py          # basic blocks
│   ├── stage1.py          # stage1 
│   ├── stage2.py          # stage2 
│   └── cascadedUNet.py    # complete cascaded u-net
├── utils/
│   ├── dataset.py         # data loading & preprocessing
│   └── dice_loss.py       # Soft Dice Loss
├── train.py               
├── predict.py             
└── evaluate.py            
```

## Environment
```bash
pip install torch nibabel scipy tqdm tensorboard
```

## Result
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
