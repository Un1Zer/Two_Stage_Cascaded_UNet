
import torch
from model.cascadedUNet import CascadedUNet

def test_shapes():
    device = 'cuda'
    model = CascadedUNet(
        in_channels=4,
        out_channels=3,
        s1_base_filters=16,
        s2_base_filters=32,
        num_groups=8,
        dropout=0.2,
        alpha=1.0,
    ).to(device)

    # 论文输入：batch=1，4通道，128^3
    x = torch.randn(1, 4, 32, 32, 32).to(device)  # 用32^3节省内存
    label = torch.randint(0, 2, (1, 3, 32, 32, 32)).float().to(device)

    # ── 训练模式 ──────────────────────────────────────────────
    model.train()
    coarse_out, out_deconv, out_interp = model(x)

    print("=== 训练模式 ===")
    print(f"输入:             {list(x.shape)}")
    print(f"Stage1 coarse:    {list(coarse_out.shape)}  (期望 [1,3,32,32,32])")
    print(f"Stage2 deconv:    {list(out_deconv.shape)}  (期望 [1,3,32,32,32])")
    print(f"Stage2 interp:    {list(out_interp.shape)}  (期望 [1,3,32,32,32])")

    # ── 推理模式 ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred = model(x)

    print("\n=== 推理模式 ===")
    print(f"最终预测:         {list(pred.shape)}   (期望 [1,3,32,32,32])")

    # ── 参数量统计 ─────────────────────────────────────────────
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"\n=== 参数量 ===")
    print(f"Stage1:           {count_params(model.stage1)/1e6:.2f}M")
    print(f"Stage2:           {count_params(model.stage2)/1e6:.2f}M")
    print(f"Total:            {count_params(model)/1e6:.2f}M")

    print("\n所有shape验证通过！")

if __name__ == '__main__':
    test_shapes()