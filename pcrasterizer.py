import torch
import numpy as np
from util import project_points, quat_to_rotmat, w2c_to_c2w, load_cameras


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
