import torch
import numpy as np
import matplotlib.pyplot as plt
from pcrasterizer import PCRasterization
from util import w2c_to_c2w, scale_intrinsics


def main():
    # 1. 加载优化好的高斯参数 (位置和SH DC系数)
    print("Loading optimized Gaussian parameters...")
    pos = torch.from_numpy(torch.load("datasets/out_bonsai/pos_param.pt")).float()
    f_dc = torch.from_numpy(torch.load("datasets/out_bonsai/f_dc.pt")).float()

    # 2. 根据SH DC系数计算颜色
    # SH DC常数 Y_0^0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
    sh_constant = 0.28209479177387814
    colors = torch.sigmoid(sh_constant * f_dc)

    # 3. 加载相机元数据及位姿 (取 cam_id = 0)
    print("Loading camera data...")
    cam_meta = np.load("datasets/out_colmap/bonsai/cam_meta.npy", allow_pickle=True).item()
    cameras = np.load("datasets/out_colmap/bonsai/cameras.npy", allow_pickle=True)

    height = cam_meta['height']
    width = cam_meta['width']
    fx = cam_meta['fx']
    fy = cam_meta['fy']
    cx = width / 2
    cy = height / 2

    cam = cameras[0]
    q_w2c = torch.tensor(cam['q']).float()
    t_w2c = torch.tensor(cam['t']).float()
    c2w = w2c_to_c2w(q_w2c, t_w2c)

    # 4. 执行光栅化渲染 (实验全分辨率和半分辨率)
    # 实验 1: 半分辨率 (更密集清晰)
    downscale = 2
    w_half = int(width / downscale)
    h_half = int(height / downscale)
    fx_h, fy_h, cx_h, cy_h = scale_intrinsics(w_half, h_half, width, height, fx, fy, cx, cy)

    print(f"Rasterizing at half resolution ({w_half}x{h_half})...")
    img_half = PCRasterization(pos, colors, h_half, w_half, fx_h, fy_h, cx_h, cy_h, camera2world=c2w)

    # 实验 2: 全分辨率
    print(f"Rasterizing at full resolution ({width}x{height})...")
    img_full = PCRasterization(pos, colors, height, width, fx, fy, cx, cy, camera2world=c2w)

    # 5. 背景颜色填充为浅灰色
    bg_color = torch.tensor([0.93, 0.94, 0.96])

    bg_mask_half = (img_half == 0.0).all(dim=-1)
    img_half[bg_mask_half] = bg_color

    bg_mask_full = (img_full == 0.0).all(dim=-1)
    img_full[bg_mask_full] = bg_color

    # 6. 保存为单独的图片
    # 保存全分辨率 (未缩放) 图片
    fig_full, ax_full = plt.subplots(figsize=(10, 6.5), facecolor='white')
    ax_full.imshow(img_full.cpu().numpy())
    ax_full.set_title(f"Bonsai Optimized Point Cloud (Original {width}x{height})", fontsize=14, fontweight='bold', pad=12)
    ax_full.axis('off')
    output_full = "bonsai_pc_original.png"
    plt.savefig(output_full, facecolor=fig_full.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Saved original resolution plot to '{output_full}'")
    plt.close(fig_full)

    # 保存半分辨率 (缩放) 图片
    fig_half, ax_half = plt.subplots(figsize=(10, 6.5), facecolor='white')
    ax_half.imshow(img_half.cpu().numpy())
    ax_half.set_title(f"Bonsai Optimized Point Cloud (Downscaled {downscale}x: {w_half}x{h_half})", fontsize=14, fontweight='bold', pad=12)
    ax_half.axis('off')
    output_half = "bonsai_pc_scaled.png"
    plt.savefig(output_half, facecolor=fig_half.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Saved scaled-down resolution plot to '{output_half}'")
    plt.close(fig_half)


if __name__ == "__main__":
    main()
