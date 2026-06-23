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
