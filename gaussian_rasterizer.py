import torch
from util import project_points


def gaussian_rasterization(pos, colors, opacity_raw, height, width, fx, fy, cx, cy, camera2world, near=2e-3, far=100, pixelGuard=64):
    # 1. 投影 3D 点到 2D 屏幕坐标
    uv, x_cam, y_cam, z_cam = project_points(pos, height, width, fx, fy, cx, cy, camera2world)
    u = uv[:, 0]
    v = uv[:, 1]

    # 2. 软边缘视锥剔除 (pixelGuard): 在图像边缘扩展 pixelGuard 像素的保护带
    mask = (u > -pixelGuard) & (u < width + pixelGuard) & \
           (v > -pixelGuard) & (v < height + pixelGuard) & \
           (z_cam > near) & (z_cam < far)

    # 3. 过滤出有效的高斯点
    u_v = u[mask]
    v_v = v[mask]
    colors_v = colors[mask]
    opacity_raw_v = opacity_raw[mask]

    # 4. 透明度重参数化 (Sigmoid & Clamp 到 0.999 避免梯度消失)
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
