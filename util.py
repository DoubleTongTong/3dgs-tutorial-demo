import torch
import numpy as np
from pathlib import Path

def quat_to_rotmat(quat):
    """
    Convert quaternion [x, y, z, w] to a 3x3 rotation matrix.
    Supports batch dimensions (..., 4) -> (..., 3, 3).
    """
    # Normalize quaternion to ensure valid rotation matrix
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    x, y, z, w = torch.unbind(quat, dim=-1)

    # Precompute products
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    # Calculate R components
    r00 = 1.0 - 2.0 * (y2 + z2)
    r01 = 2.0 * (xy - zw)
    r02 = 2.0 * (xz + yw)

    r10 = 2.0 * (xy + zw)
    r11 = 1.0 - 2.0 * (x2 + z2)
    r12 = 2.0 * (yz - xw)

    r20 = 2.0 * (xz - yw)
    r21 = 2.0 * (yz + xw)
    r22 = 1.0 - 2.0 * (x2 + y2)

    # Stack along last dimension and reshape
    rot_matrix = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    rot_matrix = rot_matrix.reshape(*quat.shape[:-1], 3, 3)
    return rot_matrix

def w2c_to_c2w(q_w2c, t_w2c):
    """
    Convert world-to-camera parameters to camera-to-world matrix.
    q_w2c: quaternion of shape (..., 4)
    t_w2c: translation of shape (..., 3)
    Returns: c2w matrix of shape (..., 4, 4)
    """
    R_w2c = quat_to_rotmat(q_w2c)

    # R_c2w = R_w2c.T (transpose the rotation matrix)
    R_c2w = R_w2c.transpose(-1, -2)

    # t_c2w = -R_c2w @ t_w2c
    t_c2w = -torch.matmul(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)

    # Construct 4x4 matrix
    shape = q_w2c.shape[:-1]
    if len(shape) == 0:
        c2w = torch.eye(4, dtype=q_w2c.dtype, device=q_w2c.device)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
    else:
        c2w = torch.eye(4, dtype=q_w2c.dtype, device=q_w2c.device).repeat(*shape, 1, 1)
        c2w[..., :3, :3] = R_c2w
        c2w[..., :3, 3] = t_c2w

    return c2w

def load_cameras(cameras_path, images_root, device='cpu', dtype=torch.float32):
    cameras_data = np.load(cameras_path, allow_pickle=True)
    cams = sorted(cameras_data, key=lambda x: x['id'])

    camera_to_worlds = []
    image_paths = []
    images_root = Path(images_root)

    for cam in cams:
        q = torch.tensor(cam['q'], device=device, dtype=dtype)
        t = torch.tensor(cam['t'], device=device, dtype=dtype)
        c2w = w2c_to_c2w(q, t)

        camera_to_worlds.append(c2w)
        image_paths.append(images_root / cam['name'])

    return camera_to_worlds, image_paths

def project_points(PC, height, width, fx, fy, cx, cy, camera2world):
    """
    将三维点投影到二维屏幕坐标系。
    根据针孔相机模型公式：
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    """
    # 1. 从 camera2world 提取旋转 R 和平移 t
    R = camera2world[:3, :3]
    t = camera2world[:3, 3]

    # 2. 计算 R 的转置
    R_t = R.t()

    # 3. 计算世界到相机转换的平移向量：t_w2c = -R_t @ t
    t_w2c = -R_t @ t

    # 4. 构造世界到相机的变换矩阵 world2camera
    world2camera = torch.eye(4, dtype=camera2world.dtype, device=camera2world.device)
    world2camera[:3, :3] = R_t
    world2camera[:3, 3] = t_w2c

    # 5. 点云拼上最后一列 1
    ones = torch.ones((PC.shape[0], 1), dtype=PC.dtype, device=PC.device)
    PC_homo = torch.cat([PC, ones], dim=1)

    # 6. 利用变换矩阵变换点云
    PC_cam = (world2camera @ PC_homo.t()).t()[:, :3]

    x_cam = PC_cam[:, 0]
    y_cam = PC_cam[:, 1]
    z_cam = PC_cam[:, 2]

    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

    uv = torch.stack([u, v], dim=-1)
    return uv, x_cam, y_cam, z_cam

def scale_intrinsics(W_target, H_target, W_source, H_source, fx, fy, cx, cy):
    scale_x = W_target / W_source
    scale_y = H_target / H_source
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y

def inverse_2x2(m, eps=1e-12):
    """
    计算批量 2x2 矩阵的逆矩阵。
    """
    a = m[:, 0, 0]
    b = m[:, 0, 1]
    c = m[:, 1, 0]
    d = m[:, 1, 1]

    det = a * d - b * c
    safe_det = torch.clamp(det, min=eps)

    inverse = torch.empty_like(m)
    inverse[:, 0, 0] = d / safe_det
    inverse[:, 0, 1] = -b / safe_det
    inverse[:, 1, 0] = -c / safe_det
    inverse[:, 1, 1] = a / safe_det

    return inverse
