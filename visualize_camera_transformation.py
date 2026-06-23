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

def generate_sphere_point_cloud(N=30000):
    """
    Generate a 3D sphere point cloud.
    """
    theta = torch.rand(N) * 2 * np.pi
    phi = torch.acos(2 * torch.rand(N) - 1)
    r = 3.0 + 0.25 * torch.randn(N)

    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    # Place the sphere at (0, 0, 10) in World Space
    PC = torch.stack([x, y, z + 10.0], dim=-1)

    # Color based on original depth
    cmap = plt.get_cmap('plasma')
    z_norm = (z - z.min()) / (z.max() - z.min())
    colors_np = cmap(z_norm.numpy())[:, :3]
    PCColor = torch.from_numpy(colors_np).float()

    return PC, PCColor

def main():
    height, width = 600, 900
    cx = width / 2
    cy = height / 2

    # 1. Generate 3D Point Cloud in World Coordinates
    PC, PCColor = generate_sphere_point_cloud(N=30000)

    # Create matplotlib figure with clean white background (Academic Style)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='white')

    # Experiment 1: Camera at World Origin (Baseline)
    c2w_1 = torch.eye(4)
    img1 = PCRasterization(PC, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy, camera2world=c2w_1)

    # Experiment 2: Camera Translate Right (X += 2.0) -> Object shifts Left
    c2w_2 = torch.eye(4)
    c2w_2[0, 3] = 2.0
    img2 = PCRasterization(PC, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy, camera2world=c2w_2)

    # Experiment 3: Camera Translate Up (Y += 1.5) -> Object shifts Down
    c2w_3 = torch.eye(4)
    c2w_3[1, 3] = 1.5
    img3 = PCRasterization(PC, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy, camera2world=c2w_3)

    # Experiment 4: Camera Yaw (Rotate 15 degrees around Y-axis) -> Object rotates/shifts Left
    c2w_4 = torch.eye(4)
    theta = 15.0 * np.pi / 180.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Y-axis Rotation Matrix:
    # [ cos  0  sin ]
    # [  0   1   0  ]
    # [ -sin 0  cos ]
    c2w_4[0, 0] = cos_t
    c2w_4[0, 2] = sin_t
    c2w_4[2, 0] = -sin_t
    c2w_4[2, 2] = cos_t

    # Also translate camera slightly to keep object nicely in view
    c2w_4[0, 3] = 1.5
    img4 = PCRasterization(PC, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy, camera2world=c2w_4)

    images = [img1, img2, img3, img4]
    bg_color = torch.tensor([0.93, 0.94, 0.96])
    for img in images:
        bg_mask = (img == 0.0).all(dim=-1)
        img[bg_mask] = bg_color

    titles = [
        "(a) Baseline (Camera at Origin)",
        "(b) Camera Shifted Right ($X_{\\mathrm{cam}} = +2.0$)",
        "(c) Camera Shifted Up ($Y_{\\mathrm{cam}} = +1.5$)",
        "(d) Camera Rotated (Yaw $15^\\circ$ + Translate)"
    ]

    for ax, img, title in zip(axes.flat, images, titles):
        img_np = img.cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(title, color='#000000', fontsize=13, fontweight='bold', pad=10)
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_filename = "camera_transformation_experiments.png"
    plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Camera transformation verification plot saved as '{output_filename}'")

if __name__ == "__main__":
    main()
