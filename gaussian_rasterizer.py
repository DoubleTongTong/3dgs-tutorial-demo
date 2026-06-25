import torch
from util import project_points, inverse_2x2


def gaussian_rasterization(pos, colors, opacity_raw, height, width, fx, fy, cx, cy, camera2world, sigma=None, near=2e-3, far=100, pixelGuard=64, tile_size=16, min_conic=1e-6, chi_square_clip=9.21, alpha_max=0.99, alpha_cutoff=1.0/255.0):
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

    # 计算高斯球在屏幕上的包围盒 (AABB) 与筛选
    evals_v = evals[keep]
    evals_sorted = evals_v[order]
    major_variance = evals_sorted[:, 1].clamp(min=1e-12, max=1e4)
    radius = torch.ceil(3.0 * torch.sqrt(major_variance)).to(torch.int64)

    u_min = torch.floor(u_sorted - radius)
    u_max = torch.ceil(u_sorted + radius)
    v_min = torch.floor(v_sorted - radius)
    v_max = torch.ceil(v_sorted + radius)

    onscreen = (u_max > 0) & (u_min < width) & (v_max > 0) & (v_min < height)
    if not onscreen.any():
        raise Exception("没有高斯球在屏幕范围内！(No Gaussians onscreen!)")

    u_sorted = u_sorted[onscreen]
    v_sorted = v_sorted[onscreen]
    colors_sorted = colors_sorted[onscreen]
    opacity_sorted = opacity_sorted[onscreen]
    sigma_camera_sorted = sigma_camera_sorted[onscreen]

    u_min = u_min[onscreen].clamp(0, width - 1)
    u_max = u_max[onscreen].clamp(0, width - 1)
    v_min = v_min[onscreen].clamp(0, height - 1)
    v_max = v_max[onscreen].clamp(0, height - 1)

    # 从像素坐标转换到 Tile 索引并转换为 int64
    u_min_tile = (u_min / tile_size).to(torch.int64)
    u_max_tile = (u_max / tile_size).to(torch.int64)
    v_min_tile = (v_min / tile_size).to(torch.int64)
    v_max_tile = (v_max / tile_size).to(torch.int64)

    # 计算每个高斯在水平(u)和垂直(v)方向跨越的 Tile 数量 (+1 避免 Fencepost 误差)
    n_u = u_max_tile - u_min_tile + 1
    n_v = v_max_tile - v_min_tile + 1

    # 获取全局单个高斯跨越的最大 Tile 数 (用于统计或后续阶段验证)
    nu_max_item = int(n_u.max().item())
    nv_max_item = int(n_v.max().item())

    # 计算高斯与 Tile 的相交映射
    device = pos.device
    span_indices_u = torch.arange(nu_max_item, device=device, dtype=torch.int64)
    span_indices_v = torch.arange(nv_max_item, device=device, dtype=torch.int64)

    # 起始点加上跨度，得到每个高斯在每个跨度上的 Tile 坐标 (未过滤，有过度填充)
    tile_u = u_min_tile[:, None] + span_indices_u[None, :]  # Shape: [N, nu_max_item]
    tile_v = v_min_tile[:, None] + span_indices_v[None, :]  # Shape: [N, nv_max_item]

    # 创建掩码，以过滤掉超出该高斯实际跨度 (n_u, n_v) 的 Tile
    mask_u = span_indices_u[None, :] < n_u[:, None]  # Shape: [N, nu_max_item]
    mask_v = span_indices_v[None, :] < n_v[:, None]  # Shape: [N, nv_max_item]
    mask = mask_u[:, :, None] & mask_v[:, None, :]  # Shape: [N, nu_max_item, nv_max_item]

    # 构建高斯 ID 数组 (每个高斯按相交的 Tile 数量进行重复)
    num_tiles_per_gaussian = n_u * n_v                                      # Shape: [N]
    num_gaussians = u_min_tile.shape[0]
    base_ids = torch.arange(num_gaussians, dtype=torch.int64, device=device) # Shape: [N]
    gaussian_ids = torch.repeat_interleave(base_ids, num_tiles_per_gaussian) # Shape: [M] (M 为所有高斯覆盖 Tile 的总数)

    # 二维 Tile 坐标的“扁平化” (Flatten)
    num_tiles_u = (width + tile_size - 1) // tile_size
    tile_u_grid = tile_u[:, :, None].expand(-1, -1, nv_max_item)             # Shape: [N, nu_max_item, nv_max_item]
    tile_v_grid = tile_v[:, None, :].expand(-1, nu_max_item, -1)             # Shape: [N, nu_max_item, nv_max_item]
    tile_u_flat = tile_u_grid[mask]                                          # Shape: [M]
    tile_v_flat = tile_v_grid[mask]                                          # Shape: [M]
    flat_tile_id = tile_v_flat * num_tiles_u + tile_u_flat                   # Shape: [M]

    # 7. 计算 2D 逆协方差矩阵
    inv_cov = inverse_2x2(sigma_camera_sorted)
    inv_cov[:, 0, 0] = torch.clamp(inv_cov[:, 0, 0], min=min_conic)
    inv_cov[:, 1, 1] = torch.clamp(inv_cov[:, 1, 1], min=min_conic)

    # 8. 基于 Tile 循环迭代渲染并生成像素坐标网格
    # 声明并初始化零画布并获取其一维展平引用
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=pos.device)
    image_flat = image.view(-1, 3)

    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size

    for tyi in range(num_tiles_y):
        for txi in range(num_tiles_x):
            # 计算当前 Tile 的像素边界 (X0, Y0 为左上角，X1, Y1 为右下角)
            x0 = txi * tile_size
            y0 = tyi * tile_size
            x1 = min((txi + 1) * tile_size, width)
            y1 = min((tyi + 1) * tile_size, height)

            # 生成 X 和 Y 方向的一维坐标序列
            xs = torch.arange(x0, x1, dtype=pos.dtype, device=pos.device)
            ys = torch.arange(y0, y1, dtype=pos.dtype, device=pos.device)

            # 生成 Tile 内的二维像素网格 (使用 xy 索引模式，以符合图像直觉)
            px, py = torch.meshgrid(xs, ys, indexing='xy')

            # 重塑为一维数组 (px_u 和 px_v 分别代表该 Tile 内部的横、纵坐标序列)
            px_u = px.reshape(-1)
            px_v = py.reshape(-1)

            # 计算在全局一维图像数组中的像素索引 (Y * Width + X)
            pixel_idx_1D = (px_v * width + px_u).to(torch.int64)

            # 筛选与当前 Tile 相交的高斯 (使用更直观的 Tile 索引范围判断)
            ids_tile = torch.where(
                (u_min_tile <= txi) & (u_max_tile >= txi) &
                (v_min_tile <= tyi) & (v_max_tile >= tyi)
            )[0]

            if len(ids_tile) == 0:
                continue

            # 提取对应高斯的属性
            u_tile = u_sorted[ids_tile]
            v_tile = v_sorted[ids_tile]
            colors_tile = colors_sorted[ids_tile]
            opacity_tile = opacity_sorted[ids_tile]
            inv_cov_tile = inv_cov[ids_tile]

            # 计算像素点到高斯中心的距离 (du, dv)
            du = px_u.unsqueeze(0) - u_tile.unsqueeze(1)  # (N, P)
            dv = px_v.unsqueeze(0) - v_tile.unsqueeze(1)  # (N, P)

            # 计算 2D 高斯密度与透明度 alpha (Equation 2 核心物理公式实现)
            a11 = inv_cov_tile[:, 0, 0].unsqueeze(1)  # (N, 1)
            a12 = inv_cov_tile[:, 0, 1].unsqueeze(1)  # (N, 1)
            a22 = inv_cov_tile[:, 1, 1].unsqueeze(1)  # (N, 1)

            # 计算 Q 值（马氏距离的平方）
            Q = a11 * du**2 + 2.0 * a12 * du * dv + a22 * dv**2  # (N, P)

            # 限制 Q 的上限以避免数值爆炸
            Q = torch.clamp(Q, max=chi_square_clip)

            # 99% 置信区间裁剪
            inside = Q <= chi_square_clip
            G = torch.exp(-0.5 * Q)  # (N, P)
            G = torch.where(inside, G, 0.0)

            alpha = opacity_tile.unsqueeze(1) * G  # (N, P)
            alpha = torch.clamp(alpha, max=alpha_max)
            alpha = torch.where(alpha >= alpha_cutoff, alpha, 0.0)

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

            # 写入画布对应的像素索引处
            image_flat[pixel_idx_1D] = tile_colors

    return image
