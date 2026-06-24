import torch
from util import project_points


def gaussian_rasterization(pos, colors, opacity_raw, height, width, fx, fy, cx, cy, camera2world, sigma=None, near=2e-3, far=100, pixelGuard=64):
    N = pos.shape[0]
    if sigma is None:
        sigma = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0).repeat(N, 1, 1)

    # 1. 投影 3D 点到 2D 屏幕坐标
    uv, x_cam, y_cam, z_cam = project_points(pos, height, width, fx, fy, cx, cy, camera2world)
    u = uv[:, 0]
    v = uv[:, 1]

    # 2. 投影 3D 协方差到 2D 图像平面 (Equation 5: Sigma' = J * W * Sigma * W^T * J^T)
    # W: 从世界坐标系到相机坐标系的变换矩阵
    W = camera2world[:3, :3].T

    # 构建雅可比矩阵 J
    J = torch.zeros((N, 2, 3), device=pos.device, dtype=pos.dtype)
    J[:, 0, 0] = fx / z_cam
    J[:, 1, 1] = fy / z_cam
    J[:, 0, 2] = -(fx * x_cam) / (z_cam ** 2)
    J[:, 1, 2] = -(fy * y_cam) / (z_cam ** 2)

    # 矩阵乘法
    TMP = W.unsqueeze(0) @ sigma @ W.unsqueeze(0).transpose(1, 2)
    sigma_camera = J @ TMP @ J.transpose(1, 2)

    # 3. 软边缘视锥剔除 (pixelGuard): 在图像边缘扩展 pixelGuard 像素的保护带
    mask = (u > -pixelGuard) & (u < width + pixelGuard) & \
           (v > -pixelGuard) & (v < height + pixelGuard) & \
           (z_cam > near) & (z_cam < far)

    # 4. 过滤出有效的高斯点
    u_v = u[mask]
    v_v = v[mask]
    colors_v = colors[mask]
    opacity_raw_v = opacity_raw[mask]
    sigma_camera_v = sigma_camera[mask]

    # 5. 透明度重参数化 (Sigmoid & Clamp 到 0.999 避免梯度消失)
    opacity_v = torch.sigmoid(opacity_raw_v)
    opacity_v = torch.clamp(opacity_v, min=0.0, max=0.999)

    # 5. 简单点级光栅化以供阶段验证：转换浮点像素坐标为整数索引
    u_rounded = torch.round(u_v).long()
    v_rounded = torch.round(v_v).long()

    # 只把落在屏幕内的像素绘制出来
    in_screen = (u_rounded >= 0) & (u_rounded < width) & (v_rounded >= 0) & (v_rounded < height)
    u_int = u_rounded[in_screen]
    v_int = v_rounded[in_screen]
    colors_draw = colors_v[in_screen]
    opacity_draw = opacity_v[in_screen]

    # 创建画布并向量化赋值 (颜色乘以透明度以体现其强度)
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=pos.device)
    image[v_int, u_int] = opacity_draw.unsqueeze(-1) * colors_draw

    return image
