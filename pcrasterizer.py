import torch
import numpy as np

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

def PCRasterization(PC, PCColor, height, width, fx, fy, cx, cy, camera2world, near=2e-3, far=100):
    # 1. 形状校验
    assert PC.shape == PCColor.shape, "PC and PCColor must have the same shape"

    # 2. 调用投影函数并解包
    uv, x_cam, y_cam, z_cam = project_points(PC, height, width, fx, fy, cx, cy, camera2world)
    u = uv[:, 0]
    v = uv[:, 1]

    # 3. 边界与相机系远近深度判定
    valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z_cam >= near) & (z_cam <= far)

    # 4. 过滤出在视口与远近深度平面内的有效点云坐标及颜色
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    colors_valid = PCColor[valid_mask]

    # 5. 将浮点像素坐标转换为整数像素索引
    u_rounded = torch.round(u_valid).long()
    v_rounded = torch.round(v_valid).long()

    # 6. 使用 clamp 确保整数索引严格在合法画布范围内
    u_int = torch.clamp(u_rounded, 0, width - 1)
    v_int = torch.clamp(v_rounded, 0, height - 1)

    # 7. 创建画布并进行向量化渲染赋值
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=PC.device)
    image[v_int, u_int] = colors_valid

    return image

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
