import torch
from util import project_points, inverse_2x2


def gaussian_rasterization(pos, colors, opacity_raw, height, width, fx, fy, cx, cy, camera2world, sigma=None, near=2e-3, far=100, pixelGuard=64, tile_size=16, min_conic=1e-6):
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

    # 4. 过滤与数值稳定处理
    u_v = u[mask]
    v_v = v[mask]
    z_cam_v = z_cam[mask]
    colors_v = colors[mask]
    opacity_raw_v = opacity_raw[mask]
    sigma_camera_v = sigma_camera[mask]

    # 数值稳定性技巧：强制对称与保证半正定
    sigma_camera_v = (sigma_camera_v + sigma_camera_v.transpose(1, 2)) * 0.5
    evals, evex = torch.linalg.eigh(sigma_camera_v)
    evals = torch.clamp(evals, min=1e-6, max=1e4)
    sigma_camera_v = evex @ torch.diag_embed(evals) @ evex.transpose(1, 2)

    # 异常数据过滤 (Guardrail Masking)
    flat_sigma = sigma_camera_v.reshape(sigma_camera_v.shape[0], -1)
    keep = torch.isfinite(flat_sigma).all(dim=-1)

    u_v = u_v[keep]
    v_v = v_v[keep]
    z_cam_v = z_cam_v[keep]
    colors_v = colors_v[keep]
    opacity_raw_v = opacity_raw_v[keep]
    sigma_camera_v = sigma_camera_v[keep]

    # 5. 透明度重参数化 (Sigmoid & Clamp 到 0.999 避免梯度消失)
    opacity_v = torch.sigmoid(opacity_raw_v)
    opacity_v = torch.clamp(opacity_v, min=0.0, max=0.999)

    # 6. 全局深度排序 (Global Depth Sorting)
    order = torch.argsort(z_cam_v, descending=False)
    u_sorted = u_v[order]
    v_sorted = v_v[order]
    colors_sorted = colors_v[order]
    opacity_sorted = opacity_v[order]
    sigma_camera_sorted = sigma_camera_v[order]

    # 7. 计算 2D 逆协方差矩阵
    inv_cov = inverse_2x2(sigma_camera_sorted)
    inv_cov[:, 0, 0] = torch.clamp(inv_cov[:, 0, 0], min=min_conic)
    inv_cov[:, 1, 1] = torch.clamp(inv_cov[:, 1, 1], min=min_conic)

    # 8. 筛选相交高斯与像素坐标网格 (TODO)
    # TODO: 实现基于 16x16 瓦片的循环筛选和网格生成
    # 暂定变量 ids 包含所有高斯，pixels 包含图像所有像素，作为已实现的占位代理
    ids = torch.arange(len(u_sorted), device=pos.device)

    grid_v, grid_u = torch.meshgrid(
        torch.arange(height, device=pos.device, dtype=pos.dtype),
        torch.arange(width, device=pos.device, dtype=pos.dtype),
        indexing='ij'
    )
    pixels = torch.stack([grid_u.flatten(), grid_v.flatten()], dim=-1)  # (P, 2)

    # 声明并初始化零画布
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=pos.device)

    # 构造模拟的单循环迭代 todo 列表，包含占位的高斯 IDs 和像素
    todo = [(ids, pixels)]

    for ids_tile, pixels_tile in todo:
        # 提取对应高斯的属性
        u_tile = u_sorted[ids_tile]
        v_tile = v_sorted[ids_tile]
        colors_tile = colors_sorted[ids_tile]
        opacity_tile = opacity_sorted[ids_tile]
        inv_cov_tile = inv_cov[ids_tile]

        # 计算距离 (du, dv)
        du = pixels_tile[:, 0].unsqueeze(0) - u_tile.unsqueeze(1)  # (N, P)
        dv = pixels_tile[:, 1].unsqueeze(0) - v_tile.unsqueeze(1)  # (N, P)

        # 计算 2D 高斯密度与透明度 alpha (Equation 2 核心物理公式实现)
        A = inv_cov_tile[:, 0, 0].unsqueeze(1)  # (N, 1)
        B = inv_cov_tile[:, 0, 1].unsqueeze(1)  # (N, 1)
        C = inv_cov_tile[:, 1, 1].unsqueeze(1)  # (N, 1)

        power = -0.5 * (A * du**2 + 2.0 * B * du * dv + C * dv**2)
        density = torch.exp(power)  # (N, P)
        alpha = opacity_tile.unsqueeze(1) * density  # (N, P)
        alpha = torch.clamp(alpha, max=0.999)

        # 计算累积透射率 T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
        ti = torch.cumprod(1.0 - alpha, dim=0)

        # 错位偏置一位，首位补 1
        ti = torch.cat([
            torch.ones((1, alpha.shape[1]), device=alpha.device, dtype=alpha.dtype),
            ti[:-1]
        ], dim=0)

        # 计算权重 w_i = alpha_i * T_i
        w = alpha * ti  # (N, P)

        # 混合颜色：\sum_i w_i * c_i
        tile_colors = (w.unsqueeze(-1) * colors_tile.unsqueeze(1)).sum(dim=0)  # (P, 3)

        # 转换为图像形状并写入画布
        image = tile_colors.view(height, width, 3)

    return image
