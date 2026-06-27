import numpy as np
from plyfile import PlyData
import torch
import os

def load_ply_gaussians(ply_path):
    print(f"Reading PLY file from {ply_path}...")
    plydata = PlyData.read(ply_path)

    vertex = plydata['vertex']

    # 1. 提取位置 (XYZ)
    xyz = np.stack((np.asarray(vertex['x']),
                    np.asarray(vertex['y']),
                    np.asarray(vertex['z'])), axis=-1)
    print(f"Loaded XYZ: {xyz.shape}")

    # 2. 提取不透明度 (Opacity)
    opacities = np.asarray(vertex['opacity'])
    print(f"Loaded Opacity: {opacities.shape}")

    # 3. 提取尺度缩放 (Scale)
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=-1)
    print(f"Loaded Scale: {scales.shape}")

    # 4. 提取旋转四元数 (Rotation) [格式通常为 w, x, y, z]
    rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rotations = np.stack([np.asarray(vertex[name]) for name in rot_names], axis=-1)
    print(f"Loaded Rotation (Quat): {rotations.shape}")

    # 5. 提取球谐函数系数 (Spherical Harmonics)
    f_dc = np.stack([
        np.asarray(vertex['f_dc_0']),
        np.asarray(vertex['f_dc_1']),
        np.asarray(vertex['f_dc_2'])
    ], axis=-1)
    print(f"Loaded f_dc (0-degree SH): {f_dc.shape}")

    f_rest_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
    f_rest = np.stack([np.asarray(vertex[name]) for name in f_rest_names], axis=-1)
    print(f"Loaded f_rest (1-3 degree SH): {f_rest.shape}")

    return {
        "pos": xyz,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "alpha_raw": opacities,
        "scale_raw": scales,
        "rot_raw": rotations
    }

def main():
    ply_path = "datasets/room.ply"
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Cannot find ply file at '{ply_path}'")

    data = load_ply_gaussians(ply_path)

    out_dir = "datasets/out_room"
    os.makedirs(out_dir, exist_ok=True)

    for key, val in data.items():
        # 如果 key 里已经有 param，则直接保存为 key.pt，否则保存为 key_param.pt
        name = f"{key}_param.pt" if "param" not in key else f"{key}.pt"
        save_path = os.path.join(out_dir, name)
        print(f"Saving {key} to {save_path}...")
        torch.save(val, save_path)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
