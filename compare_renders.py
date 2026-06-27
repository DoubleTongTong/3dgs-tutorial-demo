import os
import torch
import numpy as np
from PIL import Image
from util import w2c_to_c2w, compute_3d_covariance, evaluate_sh
from gaussian_rasterizer import gaussian_rasterization
from load_ply import load_ply_gaussians
import gsplat

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 实时直接解析 room 数据集
    print("Loading room dataset directly from datasets/room.ply...")
    ply_data = load_ply_gaussians("datasets/room.ply")
    pos = ply_data["pos"]
    f_dc = ply_data["f_dc"]
    f_rest = ply_data["f_rest"]
    alpha_raw = ply_data["alpha_raw"]
    rot_raw = ply_data["rot_raw"]
    scale_raw = ply_data["scale_raw"]

    # 转换为 float 并送至 GPU
    pos = torch.from_numpy(pos).float().to(device)
    f_dc = torch.from_numpy(f_dc).float().to(device)
    f_rest = torch.from_numpy(f_rest).float().to(device)
    alpha_raw = torch.from_numpy(alpha_raw).float().to(device)
    rot_raw = torch.from_numpy(rot_raw).float().to(device)
    scale_raw = torch.from_numpy(scale_raw).float().to(device)

    # 2. 构造一个虚拟相机
    center = pos.mean(dim=0)
    print(f"Scene center: {center.cpu().numpy()}")

    # 虚拟一个相机位置 (c2w)：放置在中心点外围，往前往下看
    c2w = torch.eye(4, device=device)
    c2w[:3, 3] = center + torch.tensor([0.0, -0.5, -1.5], device=device) # 往前往下移动相机

    # 焦距与视口大小
    height, width = 480, 640
    fx, fy = 500.0, 500.0
    cx, cy = width / 2.0, height / 2.0

    # 3. 基于当前相机的视角计算 SH 颜色
    print("Evaluating SH colors...")
    colors = evaluate_sh(f_dc, f_rest, pos, c2w)

    # 4. 执行我们当前的 Tile 渲染器进行渲染
    print("Rendering with OUR Tile Rasterizer...")
    sigma = compute_3d_covariance(scale_raw, rot_raw)
    img_ours = gaussian_rasterization(
        pos, colors, alpha_raw, height, width, fx, fy, cx, cy,
        camera2world=c2w, sigma=sigma, near=2e-3, far=100
    )

    bg_color = torch.tensor([0.93, 0.94, 0.96], device=device)
    bg_mask_ours = (img_ours == 0.0).all(dim=-1)
    img_ours[bg_mask_ours] = bg_color

    img_ours_np = (img_ours.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img_ours_np).save("room_render_ours.png")
    print("Saved our rendering to 'room_render_ours.png'")
    print(f"img_ours raw range: min {img_ours.min().item():.4f}, max {img_ours.max().item():.4f}, mean {img_ours.mean().item():.4f}")

    # 5. 执行官方 gsplat 渲染器进行对比 (Direct Mode)
    print("Rendering with official gsplat rasterizer (Direct Mode)...")
    opacities_gsplat = torch.sigmoid(alpha_raw) # shape: (N,)
    scales_gsplat = torch.exp(scale_raw) # shape: (N, 3)
    quats_gsplat = rot_raw / torch.norm(rot_raw, dim=-1, keepdim=True)

    # viewmats 为 W2C 矩阵，shape (1, 4, 4)
    w2c = c2w.inverse().unsqueeze(0)

    # Ks 内参矩阵，shape (1, 3, 3)
    Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device).unsqueeze(0)

    out_color, out_alpha, _ = gsplat.rasterization(
        means=pos,
        quats=quats_gsplat,
        scales=scales_gsplat,
        opacities=opacities_gsplat,
        colors=colors,
        viewmats=w2c,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=2e-3,
        far_plane=100.0,
        eps2d=0.3
    )

    img_gsplat = out_color[0]
    bg_mask_gsplat = (img_gsplat == 0.0).all(dim=-1)
    img_gsplat[bg_mask_gsplat] = bg_color

    img_gsplat_np = (img_gsplat.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img_gsplat_np).save("room_render_gsplat.png")
    print("Saved gsplat rendering to 'room_render_gsplat.png'")
    print(f"img_gsplat raw range: min {img_gsplat.min().item():.4f}, max {img_gsplat.max().item():.4f}, mean {img_gsplat.mean().item():.4f}")

    mae_gsplat = np.mean(np.abs(img_ours_np.astype(float) - img_gsplat_np.astype(float)))
    print(f"MAE between Our Tile Rasterizer and official gsplat: {mae_gsplat:.4f}")

    # 6. 使用 gsplat 官方的 CUDA 球谐函数评估（SH Mode）进行彩色渲染
    print("Rendering with official gsplat rasterizer using SH Mode...")
    sh_coeffs = torch.cat([f_dc.unsqueeze(1), f_rest.reshape(-1, 15, 3)], dim=1)

    out_color_sh, _, _ = gsplat.rasterization(
        means=pos,
        quats=quats_gsplat,
        scales=scales_gsplat,
        opacities=opacities_gsplat,
        colors=sh_coeffs,
        viewmats=w2c,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=2e-3,
        far_plane=100.0,
        sh_degree=3,
        eps2d=0.3
    )

    img_gsplat_sh = out_color_sh[0]
    bg_mask_gsplat_sh = (img_gsplat_sh == 0.0).all(dim=-1)
    img_gsplat_sh[bg_mask_gsplat_sh] = bg_color

    img_gsplat_sh_np = (img_gsplat_sh.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img_gsplat_sh_np).save("room_render_gsplat_sh.png")
    print("Saved gsplat SH Mode rendering to 'room_render_gsplat_sh.png'")
    print(f"img_gsplat_sh raw range: min {img_gsplat_sh.min().item():.4f}, max {img_gsplat_sh.max().item():.4f}, mean {img_gsplat_sh.mean().item():.4f}")

    mae_gsplat_self = np.mean(np.abs(img_gsplat_np.astype(float) - img_gsplat_sh_np.astype(float)))
    print(f"MAE between our python evaluate_sh and official gsplat CUDA SH: {mae_gsplat_self:.4f}")

if __name__ == "__main__":
    main()
