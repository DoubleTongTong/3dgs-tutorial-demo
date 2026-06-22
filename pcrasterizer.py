import torch
import numpy as np

def project_points(PC, height, width, fx, fy, cx, cy):
    """
    将三维点投影到二维屏幕坐标系。
    根据针孔相机模型公式：
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    """
    x_cam = PC[:, 0]
    y_cam = PC[:, 1]
    z_cam = PC[:, 2]

    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

    uv = torch.stack([u, v], dim=-1)
    return uv, x_cam, y_cam, z_cam

def PCRasterization(PC, PCColor, height, width, fx, fy, cx, cy, near=2e-3, far=100):
    # 1. 形状校验
    assert PC.shape == PCColor.shape, "PC and PCColor must have the same shape"

    # 2. 调用投影函数并解包
    uv, x_cam, y_cam, z_cam = project_points(PC, height, width, fx, fy, cx, cy)
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
