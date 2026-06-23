import torch
import numpy as np
import matplotlib.pyplot as plt
from pcrasterizer import PCRasterization

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

    # 3. Create dummy camera pose (c2w = Identity)
    c2w = torch.eye(4)

    # 4. Perform rasterization
    print("Running PCRasterization with dummy camera pose...")
    img = PCRasterization(PC, PCColor, height, width, fx, fy, cx, cy, camera2world=c2w)

    # 5. Apply neutral light gray background (post-processing)
    # The default background is black (all zeros)
    bg_color = torch.tensor([0.93, 0.94, 0.96])
    bg_mask = (img == 0.0).all(dim=-1)
    img[bg_mask] = bg_color

    # 6. Plotting and saving
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')
    img_np = img.cpu().numpy()
    ax.imshow(img_np)
    ax.set_title("Bonsai Point Cloud Rasterization (Dummy Camera Pose)", color='#000000', fontsize=14, fontweight='bold', pad=12)
    ax.axis('on')
    ax.set_xticks([])
    ax.set_yticks([])

    output_filename = "real_data_dummy_rasterization.png"
    plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Rasterization plot saved as '{output_filename}'")

if __name__ == "__main__":
    main()
