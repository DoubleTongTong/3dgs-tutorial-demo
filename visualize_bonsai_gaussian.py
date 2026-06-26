import torch
import numpy as np
import random
from PIL import Image
from gaussian_rasterizer import gaussian_rasterization
from util import w2c_to_c2w, compute_3d_covariance


def render_and_save_view(pos, colors, alpha_raw, sigma, height, width, fx, fy, cx, cy, c2w, bg_color, output_path):
    """
    辅助函数：执行 3D 轴对称高斯光栅化渲染，并将张量直接以像素完美（pixel-perfect）的格式保存为图片，避免 matplotlib 重采样带来的伪影
    """
    img = gaussian_rasterization(
        pos, colors, alpha_raw, height, width, fx, fy, cx, cy,
        camera2world=c2w, sigma=sigma, near=2e-3, far=100
    )
    # 背景颜色填充为浅灰色
    bg_mask = (img == 0.0).all(dim=-1)
    img[bg_mask] = bg_color

    # [0, 1] 范围的 Float 转换成 [0, 255] uint8 并保存
    img_np = (img.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img_np).save(output_path)
    print(f"Saved pixel-perfect rendering to '{output_path}'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载优化好的高斯参数 (位置、颜色、缩放、旋转和不透明度)
    print("Loading optimized Gaussian parameters...")
    pos = torch.from_numpy(torch.load("datasets/out_bonsai/pos_param.pt")).to(device).float()
    f_dc = torch.from_numpy(torch.load("datasets/out_bonsai/f_dc.pt")).to(device).float()
    alpha_raw = torch.from_numpy(torch.load("datasets/out_bonsai/alpha_raw_param.pt")).to(device).float()
    rot_raw = torch.from_numpy(torch.load("datasets/out_bonsai/rot_raw.pt")).to(device).float()
    scale_raw = torch.from_numpy(torch.load("datasets/out_bonsai/scale_raw.pt")).to(device).float()

    # 2. 计算 3D 协方差矩阵 (sigma)
    print("Computing 3D covariance matrices (sigma)...")
    sigma = compute_3d_covariance(scale_raw, rot_raw)

    # 3. 计算高斯球颜色 (SH DC 系数转 RGB)
    # SH DC常数 Y_0^0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
    sh_constant = 0.28209479177387814
    colors = torch.sigmoid(sh_constant * f_dc)

    # 4. 加载相机元数据及位姿
    print("Loading camera data...")
    cam_meta = np.load("datasets/out_colmap/bonsai/cam_meta.npy", allow_pickle=True).item()
    cameras = np.load("datasets/out_colmap/bonsai/cameras.npy", allow_pickle=True)

    height = cam_meta['height']
    width = cam_meta['width']
    fx = cam_meta['fx']
    fy = cam_meta['fy']
    cx = width / 2
    cy = height / 2

    bg_color = torch.tensor([0.93, 0.94, 0.96], device=device)

    # 5. 渲染默认视角 (cam_id = 0)
    print(f"Rasterizing 3DGS at original resolution ({width}x{height}) for camera 0...")
    q_w2c_0 = torch.tensor(cameras[0]['q'], device=device).float()
    t_w2c_0 = torch.tensor(cameras[0]['t'], device=device).float()
    c2w_0 = w2c_to_c2w(q_w2c_0, t_w2c_0)

    render_and_save_view(
        pos, colors, alpha_raw, sigma, height, width, fx, fy, cx, cy, c2w_0, bg_color,
        "bonsai_3dgs_view_cam_0.png"
    )

    # 6. 随机选择 3 个其他相机视角进行渲染
    random.seed(42)  # 设定随机种子以保证结果可复现
    random_cam_indices = random.sample(range(1, len(cameras)), 3)
    print(f"Randomly selected camera indices for extra views: {random_cam_indices}")

    for idx in random_cam_indices:
        cam = cameras[idx]
        print(f"Rendering extra view for camera {idx} ({cam['name']})...")
        q_w2c = torch.tensor(cam['q'], device=device).float()
        t_w2c = torch.tensor(cam['t'], device=device).float()
        c2w = w2c_to_c2w(q_w2c, t_w2c)

        output_path = f"bonsai_3dgs_view_cam_{idx}.png"
        render_and_save_view(
            pos, colors, alpha_raw, sigma, height, width, fx, fy, cx, cy, c2w, bg_color,
            output_path
        )


if __name__ == "__main__":
    main()
