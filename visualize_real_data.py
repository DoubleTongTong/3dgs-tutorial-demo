import torch
import numpy as np
import matplotlib.pyplot as plt
from pcrasterizer import PCRasterization, w2c_to_c2w

def scale_intrinsics(W_target, H_target, W_source, H_source, fx, fy, cx, cy):
    # 1. 计算宽高的缩放比例 (Scale Factors)
    scale_x = W_target / W_source
    scale_y = H_target / H_source

    # 2. 对焦距进行等比例缩放
    scaled_fx = fx * scale_x
    scaled_fy = fy * scale_y

    # 3. 对光心（主点）位置进行等比例缩放
    scaled_cx = cx * scale_x
    scaled_cy = cy * scale_y

    return scaled_fx, scaled_fy, scaled_cx, scaled_cy

# Set academic plotting style parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'

def main():
    # 1. Load point cloud
    pc_path = "datasets/out_colmap/bonsai/point_cloud.npy"
    print(f"Loading point cloud from {pc_path}...")
    pc_data = np.load(pc_path)  # shape: (N, 6)

    # Extract XYZ and RGB
    xyz = pc_data[:, :3]   # shape: (N, 3)
    rgb = pc_data[:, 3:6]  # shape: (N, 3)

    # Convert to PyTorch tensors
    PC = torch.from_numpy(xyz).float()
    PCColor = torch.from_numpy(rgb).float() / 255.0  # Normalize to [0, 1]

    # 2. Load camera metadata
    meta_path = "datasets/out_colmap/bonsai/cam_meta.npy"
    print(f"Loading camera metadata from {meta_path}...")
    cam_meta = np.load(meta_path, allow_pickle=True).item()
    print("Camera metadata:", cam_meta)

    height = cam_meta['height']
    width = cam_meta['width']
    fx = cam_meta['fx']
    fy = cam_meta['fy']
    cx = width / 2
    cy = height / 2

    # 3. Load camera poses and create camera pose
    cameras_path = "datasets/out_colmap/bonsai/cameras.npy"
    print(f"Loading camera parameters from {cameras_path}...")
    cameras = np.load(cameras_path, allow_pickle=True)

    # Select the first camera pose (cam_id = 0)
    cam_id = 0
    cam = cameras[cam_id]
    print(f"Using camera {cam_id} ({cam['name']}): q={cam['q']}, t={cam['t']}")

    q_w2c = torch.tensor(cam['q']).float()
    t_w2c = torch.tensor(cam['t']).float()
    c2w = w2c_to_c2w(q_w2c, t_w2c)

    # 4. Perform rasterization
    # Experiment 1: Original Resolution (Sparse)
    print("Running PCRasterization at original resolution (expecting sparse/hard to see)...")
    img_orig = PCRasterization(PC, PCColor, height, width, fx, fy, cx, cy, camera2world=c2w)

    # Experiment 2: Downscaled Resolution with scaled intrinsics (Clear)
    downscale_factor = 4
    width_target = int(width / downscale_factor)
    height_target = int(height / downscale_factor)

    scaled_fx, scaled_fy, scaled_cx, scaled_cy = scale_intrinsics(
        width_target, height_target, width, height, fx, fy, cx, cy
    )

    print(f"Running PCRasterization at downscaled resolution ({downscale_factor}x smaller: {width_target}x{height_target})...")
    img_scaled = PCRasterization(PC, PCColor, height_target, width_target, scaled_fx, scaled_fy, scaled_cx, scaled_cy, camera2world=c2w)

    # 5. Apply neutral light gray background (post-processing)
    # The default background is black (all zeros)
    bg_color = torch.tensor([0.93, 0.94, 0.96])

    bg_mask_orig = (img_orig == 0.0).all(dim=-1)
    img_orig[bg_mask_orig] = bg_color

    bg_mask_scaled = (img_scaled == 0.0).all(dim=-1)
    img_scaled[bg_mask_scaled] = bg_color

    # 6. Plotting and saving
    # 6.1 Save the original high-resolution image
    fig_orig, ax_orig = plt.subplots(figsize=(10, 6.5), facecolor='white')
    img_orig_np = img_orig.cpu().numpy()
    ax_orig.imshow(img_orig_np)
    ax_orig.set_title(f"Bonsai Point Cloud (Original {width}x{height})", color='#000000', fontsize=14, fontweight='bold', pad=12)
    ax_orig.axis('on')
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    orig_filename = "real_data_rasterization_original.png"
    plt.savefig(orig_filename, facecolor=fig_orig.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Original rasterization plot saved as '{orig_filename}'")
    plt.close(fig_orig)

    # 6.2 Save the downscaled image
    fig_scaled, ax_scaled = plt.subplots(figsize=(10, 6.5), facecolor='white')
    img_scaled_np = img_scaled.cpu().numpy()
    ax_scaled.imshow(img_scaled_np)
    ax_scaled.set_title(f"Bonsai Point Cloud (Downscaled {downscale_factor}x: {width_target}x{height_target})", color='#000000', fontsize=14, fontweight='bold', pad=12)
    ax_scaled.axis('on')
    ax_scaled.set_xticks([])
    ax_scaled.set_yticks([])
    scaled_filename = "real_data_rasterization_scaled.png"
    plt.savefig(scaled_filename, facecolor=fig_scaled.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Scaled-down rasterization plot saved as '{scaled_filename}'")
    plt.close(fig_scaled)

    # 6.3 Save a side-by-side comparison to clearly demonstrate the concept
    fig_comp, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')

    axes[0].imshow(img_orig_np)
    axes[0].set_title(f"Original Resolution ({width}x{height})\n(Sparse / Hard to see)", color='#000000', fontsize=12, fontweight='bold', pad=8)
    axes[0].axis('on')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(img_scaled_np)
    axes[1].set_title(f"Downscaled {downscale_factor}x ({width_target}x{height_target})\n(With Scaled Intrinsics - Clear!)", color='#000000', fontsize=12, fontweight='bold', pad=8)
    axes[1].axis('on')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig_comp.suptitle("Camera Intrinsics Scaling Experiment: Sparse vs. Dense Point Cloud", color='#000000', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    comp_filename = "real_data_rasterization_comparison.png"
    plt.savefig(comp_filename, facecolor=fig_comp.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Comparison plot saved as '{comp_filename}'")
    plt.close(fig_comp)

if __name__ == "__main__":
    main()
