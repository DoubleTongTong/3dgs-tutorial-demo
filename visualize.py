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

def generate_sphere_point_cloud(N=20000):
    """
    Generate a 3D sphere point cloud.
    """
    # Sample points on a sphere shell
    theta = torch.rand(N) * 2 * np.pi
    phi = torch.acos(2 * torch.rand(N) - 1)

    # Radius with some thickness/noise
    r = 3.0 + 0.25 * torch.randn(N)

    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    PC = torch.stack([x, y, z], dim=-1)

    # Map depth (Z coordinate) to a standard scientific colormap (e.g., 'plasma')
    # This represents distance/depth in a professional, high-contrast manner.
    cmap = plt.get_cmap('plasma')
    z_norm = (z - z.min()) / (z.max() - z.min())
    colors_np = cmap(z_norm.numpy())[:, :3]
    PCColor = torch.from_numpy(colors_np).float()

    return PC, PCColor

def main():
    # 1. Setup Canvas Dimensions
    height, width = 600, 900
    cx = width / 2
    cy = height / 2
    # 2. Generate 3D Point Cloud
    PC_base, PCColor = generate_sphere_point_cloud(N=30000)

    # Create matplotlib figure with clean white background (Academic Style)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='white')

    # Let's perform the 4 experiments from the tutorial dialogue:

    # Experiment 1: Centered closer to the camera (Z + 8.0)
    PC1 = PC_base.clone()
    PC1[:, 2] += 8.0  # Add distance along Z
    img1 = PCRasterization(PC1, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy)

    # Experiment 2: Centered further away (Z + 20.0)
    PC2 = PC_base.clone()
    PC2[:, 2] += 20.0
    img2 = PCRasterization(PC2, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy)

    # Experiment 3: Shifted along X (X + 4.0, Z + 12.0)
    PC3 = PC_base.clone()
    PC3[:, 0] += 4.0
    PC3[:, 2] += 12.0
    img3 = PCRasterization(PC3, PCColor, height, width, fx=800, fy=800, cx=cx, cy=cy)

    # Experiment 4: Zoomed in (focal length multiplied by 2.2, Z + 12.0)
    PC4 = PC_base.clone()
    PC4[:, 2] += 12.0
    img4 = PCRasterization(PC4, PCColor, height, width, fx=800 * 2.2, fy=800 * 2.2, cx=cx, cy=cy)

    # Process images to apply a neutral light gray background (post-processing within visualization).
    # This resembles professional 3D viewports (e.g. MeshLab/Blender), avoids neon "tech-style"
    # while maintaining high visibility for bright yellow/orange points in the 'plasma' colormap.
    images = [img1, img2, img3, img4]
    bg_color = torch.tensor([0.93, 0.94, 0.96])
    for img in images:
        bg_mask = (img == 0.0).all(dim=-1)
        img[bg_mask] = bg_color

    # Plot configuration
    titles = [
        "(a) Baseline Configuration ($Z_{\\mathrm{offset}} = 8.0$)",
        "(b) Increased Depth Plane ($Z_{\\mathrm{offset}} = 20.0$)",
        "(c) Off-Center Translation ($X_{\\mathrm{offset}} = 4.0$, $Z_{\\mathrm{offset}} = 12.0$)",
        "(d) Camera Magnification ($f_x, f_y$ scaled by $2.2\\times$, $Z_{\\mathrm{offset}} = 12.0$)"
    ]

    for ax, img, title in zip(axes.flat, images, titles):
        # Convert tensor to numpy for matplotlib plotting
        img_np = img.cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(title, color='#000000', fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')

        # Show a clean outer box border for each rasterized view (academic paper style)
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_filename = "rasterization_experiments.png"
    plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    print(f"Academic-style rasterization plot successfully saved as '{output_filename}'")

if __name__ == "__main__":
    main()
