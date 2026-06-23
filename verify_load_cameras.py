import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pcrasterizer import PCRasterization, load_cameras

def scale_intrinsics(W_target, H_target, W_source, H_source, fx, fy, cx, cy):
    scale_x = W_target / W_source
    scale_y = H_target / H_source
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y

def main():
    # Set random seed for reproducibility
    random.seed(42)

    scenes = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]

    for scene in scenes:
        print(f"\nProcessing scene: {scene}...")
        # 1. Paths for the current scene
        pc_path = f"datasets/out_colmap/{scene}/point_cloud.npy"
        meta_path = f"datasets/out_colmap/{scene}/cam_meta.npy"
        cameras_path = f"datasets/out_colmap/{scene}/cameras.npy"
        images_root = f"datasets/360_v2/{scene}/images_2"  # 2x downscaled images

        # 2. Load Point Cloud
        print("Loading point cloud...")
        pc_data = np.load(pc_path)
        PC = torch.from_numpy(pc_data[:, :3]).float()
        PCColor = torch.from_numpy(pc_data[:, 3:6]).float() / 255.0

        # 3. Load Camera Metadata
        print("Loading camera metadata...")
        cam_meta = np.load(meta_path, allow_pickle=True).item()
        w_source, h_source = cam_meta['width'], cam_meta['height']
        fx_source, fy_source = cam_meta['fx'], cam_meta['fy']
        cx_source, cy_source = w_source / 2.0, h_source / 2.0

        # 4. Load Cameras using load_cameras
        print("Loading cameras...")
        c2ws, image_paths = load_cameras(cameras_path, images_root)

        # 5. Select 4 random camera indices
        num_cameras = len(c2ws)
        sample_indices = random.sample(range(num_cameras), min(4, num_cameras))
        print(f"Selected camera indices for visualization: {sample_indices}")

        # 6. Plot grid: 4 rows x 2 columns (Real vs. Rasterized)
        fig, axes = plt.subplots(len(sample_indices), 2, figsize=(12, 3 * len(sample_indices)))

        # Ensure axes is 2D even if we only selected 1 camera
        if len(sample_indices) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, cam_idx in enumerate(sample_indices):
            c2w = c2ws[cam_idx]
            img_path = image_paths[cam_idx]

            # Load real image
            real_img = Image.open(img_path)
            w_target, h_target = real_img.size

            # Downscale target resolution for rasterization to make the point cloud denser and clearer
            raster_downscale = 4
            w_raster = int(w_target / raster_downscale)
            h_raster = int(h_target / raster_downscale)

            # Scale intrinsics to the smaller raster resolution
            fx, fy, cx, cy = scale_intrinsics(w_raster, h_raster, w_source, h_source, fx_source, fy_source, cx_source, cy_source)

            # Rasterize at the denser resolution
            rasterized_img = PCRasterization(PC, PCColor, h_raster, w_raster, fx, fy, cx, cy, camera2world=c2w)

            # Post-process background
            bg_color = torch.tensor([0.93, 0.94, 0.96])
            bg_mask = (rasterized_img == 0.0).all(dim=-1)
            rasterized_img[bg_mask] = bg_color
            rasterized_np = rasterized_img.numpy()

            # Display Real vs Rasterized
            axes[row_idx, 0].imshow(real_img)
            axes[row_idx, 0].set_title(f"Real Image - Cam {cam_idx}")
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(rasterized_np)
            axes[row_idx, 1].set_title(f"Rasterized Point Cloud - Cam {cam_idx}")
            axes[row_idx, 1].axis('off')

        plt.suptitle(f"Scene: {scene.upper()} - Camera Alignment Verification", fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_filename = f"load_cameras_verification_{scene}.png"
        plt.savefig(output_filename, dpi=150)
        plt.close(fig)
        print(f"Verification image saved as '{output_filename}'")

if __name__ == "__main__":
    main()
