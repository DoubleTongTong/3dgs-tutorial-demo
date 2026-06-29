import torch
from util import project_points, inverse_2x2

class RasterizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, colors, opacity_raw, height, width, fx, fy, cx, cy, camera2world, sigma, near=2e-3, far=100, pixelGuard=64, tile_size=16, min_conic=1e-6, chi_square_clip=9.21, alpha_max=0.99, alpha_cutoff=1.0/255.0):
        N = pos.shape[0]
        orig_indices = torch.arange(N, device=pos.device)

        # 1. 投影 3D 点到 2D 屏幕坐标
        uv, x_cam, y_cam, z_cam = project_points(pos, height, width, fx, fy, cx, cy, camera2world)
        u = uv[:, 0]
        v = uv[:, 1]

        # 2. 投影 3D 协方差到 2D 图像平面 (Equation 5: Sigma' = J * W * Sigma * W^T * J^T)
        W = camera2world[:3, :3].T

        J = torch.zeros((N, 2, 3), device=pos.device, dtype=pos.dtype)
        J[:, 0, 0] = fx / z_cam
        J[:, 1, 1] = fy / z_cam
        J[:, 0, 2] = -(fx * x_cam) / (z_cam ** 2)
        J[:, 1, 2] = -(fy * y_cam) / (z_cam ** 2)

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
        orig_indices_v = orig_indices[mask]

        sigma_camera_v = (sigma_camera_v + sigma_camera_v.transpose(1, 2)) * 0.5
        evals, evex = torch.linalg.eigh(sigma_camera_v)
        evals = torch.clamp(evals, min=1e-6, max=1e4)
        sigma_camera_v = evex @ torch.diag_embed(evals) @ evex.transpose(1, 2)

        flat_sigma = sigma_camera_v.reshape(sigma_camera_v.shape[0], -1)
        keep = torch.isfinite(flat_sigma).all(dim=-1)

        u_v = u_v[keep]
        v_v = v_v[keep]
        z_cam_v = z_cam_v[keep]
        colors_v = colors_v[keep]
        opacity_raw_v = opacity_raw_v[keep]
        sigma_camera_v = sigma_camera_v[keep]
        orig_indices_keep = orig_indices_v[keep]

        # 5. 透明度重参数化 (Sigmoid & Clamp)
        opacity_v = torch.sigmoid(opacity_raw_v)
        opacity_v = torch.clamp(opacity_v, min=0.0, max=0.999)

        # 6. 全局深度排序 (Global Depth Sorting)
        order = torch.argsort(z_cam_v, descending=False)
        u_sorted = u_v[order]
        v_sorted = v_v[order]
        colors_sorted = colors_v[order]
        opacity_sorted = opacity_v[order]
        sigma_camera_sorted = sigma_camera_v[order]
        orig_indices_sorted = orig_indices_keep[order]

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
        indices_onscreen = orig_indices_sorted[onscreen]

        u_min = u_min[onscreen].clamp(0, width - 1)
        u_max = u_max[onscreen].clamp(0, width - 1)
        v_min = v_min[onscreen].clamp(0, height - 1)
        v_max = v_max[onscreen].clamp(0, height - 1)

        # 从像素坐标转换到 Tile 索引并转换为 int64
        u_min_tile = (u_min / tile_size).to(torch.int64)
        u_max_tile = (u_max / tile_size).to(torch.int64)
        v_min_tile = (v_min / tile_size).to(torch.int64)
        v_max_tile = (v_max / tile_size).to(torch.int64)

        n_u = u_max_tile - u_min_tile + 1
        n_v = v_max_tile - v_min_tile + 1

        nu_max_item = int(n_u.max().item())
        nv_max_item = int(n_v.max().item())

        device = pos.device
        span_indices_u = torch.arange(nu_max_item, device=device, dtype=torch.int64)
        span_indices_v = torch.arange(nv_max_item, device=device, dtype=torch.int64)

        tile_u = u_min_tile[:, None] + span_indices_u[None, :]
        tile_v = v_min_tile[:, None] + span_indices_v[None, :]

        mask_u = span_indices_u[None, :] < n_u[:, None]
        mask_v = span_indices_v[None, :] < n_v[:, None]
        mask = mask_u[:, :, None] & mask_v[:, None, :]

        num_tiles_per_gaussian = n_u * n_v
        num_gaussians = u_min_tile.shape[0]
        base_ids = torch.arange(num_gaussians, dtype=torch.int64, device=device)
        gaussian_ids = torch.repeat_interleave(base_ids, num_tiles_per_gaussian)

        num_tiles_u = (width + tile_size - 1) // tile_size
        tile_u_grid = tile_u[:, :, None].expand(-1, -1, nv_max_item)
        tile_v_grid = tile_v[:, None, :].expand(-1, nu_max_item, -1)
        tile_u_flat = tile_u_grid[mask]
        tile_v_flat = tile_v_grid[mask]
        flat_tile_id = tile_v_flat * num_tiles_u + tile_u_flat

        M = num_gaussians + 1
        comp = flat_tile_id * M + gaussian_ids
        comp_sorted, permutation = torch.sort(comp)
        gaussian_ids_sorted = gaussian_ids[permutation]
        tile_ids_1d = torch.div(comp_sorted, M, rounding_mode='floor')

        unique_tile_ids, counts = torch.unique_consecutive(tile_ids_1d, return_counts=True)
        unique_starts = torch.zeros_like(unique_tile_ids)
        unique_starts[1:] = torch.cumsum(counts[:-1], dim=0)
        unique_ends = unique_starts + counts

        # 7. 计算 2D 逆协方差矩阵
        inv_cov = inverse_2x2(sigma_camera_sorted)
        inv_cov[:, 0, 0] = torch.clamp(inv_cov[:, 0, 0], min=min_conic)
        inv_cov[:, 1, 1] = torch.clamp(inv_cov[:, 1, 1], min=min_conic)

        # [关键准备] 将所有在 backward 阶段需要的 tensor 变量保存到 save_for_backward 中
        ctx.save_for_backward(
            pos, colors, opacity_raw, sigma,
            u_sorted, v_sorted, colors_sorted, opacity_sorted, inv_cov,
            gaussian_ids_sorted, unique_tile_ids, unique_starts, unique_ends, indices_onscreen
        )
        ctx.meta = {
            'height': height,
            'width': width,
            'num_tiles_u': num_tiles_u,
            'tile_size': tile_size,
            'min_conic': min_conic,
            'chi_square_clip': chi_square_clip,
            'alpha_max': alpha_max,
            'alpha_cutoff': alpha_cutoff
        }

        # 8. 基于 Tile 循环迭代渲染并生成像素坐标网格
        image = torch.zeros((height, width, 3), dtype=torch.float32, device=pos.device)
        image_flat = image.view(-1, 3)

        for tile_id, start, end in zip(unique_tile_ids.tolist(), unique_starts.tolist(), unique_ends.tolist()):
            txi = tile_id % num_tiles_u
            tyi = tile_id // num_tiles_u

            # 计算当前 Tile 的像素边界 (X0, Y0 为左上角，X1, Y1 为右下角)
            x0 = txi * tile_size
            y0 = tyi * tile_size
            x1 = min((txi + 1) * tile_size, width)
            y1 = min((tyi + 1) * tile_size, height)

            # 生成 X 和 Y 方向的一维坐标序列
            xs = torch.arange(x0, x1, dtype=pos.dtype, device=pos.device)
            ys = torch.arange(y0, y1, dtype=pos.dtype, device=pos.device)

            # 生成 Tile 内的二维像素网格
            px, py = torch.meshgrid(xs, ys, indexing='xy')

            # 重塑为一维数组
            px_u = px.reshape(-1)
            px_v = py.reshape(-1)

            # 计算在全局一维图像数组中的像素索引
            pixel_idx_1D = (px_v * width + px_u).to(torch.int64)

            # 获取当前 Tile 包含的高斯范围
            ids_tile = gaussian_ids_sorted[start:end]

            # 提取对应高斯的属性
            u_tile = u_sorted[ids_tile]
            v_tile = v_sorted[ids_tile]
            colors_tile = colors_sorted[ids_tile]
            opacity_tile = opacity_sorted[ids_tile]
            inv_cov_tile = inv_cov[ids_tile]

            # 计算像素点到高斯中心的距离 (du, dv)
            du = px_u.unsqueeze(0) - u_tile.unsqueeze(1)  # (N, P)
            dv = px_v.unsqueeze(0) - v_tile.unsqueeze(1)  # (N, P)

            # 计算 2D 高斯密度与透明度 alpha
            a11 = inv_cov_tile[:, 0, 0].unsqueeze(1)
            a12 = inv_cov_tile[:, 0, 1].unsqueeze(1)
            a22 = inv_cov_tile[:, 1, 1].unsqueeze(1)

            # 计算 Q 值（马氏距离的平方）
            Q = a11 * du**2 + 2.0 * a12 * du * dv + a22 * dv**2  # (N, P)

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

    @staticmethod
    def backward(ctx, grad_out):
        # 1. 提取前向传播保存的张量数据与元数据
        # grad_out 原始形状: (height, width, 3)
        (
            pos, colors, opacity_raw, sigma,
            u_sorted, v_sorted, colors_sorted, opacity_sorted, inv_cov,
            gaussian_ids_sorted, unique_tile_ids, unique_starts, unique_ends, indices_onscreen
        ) = ctx.saved_tensors

        height = ctx.meta['height']
        width = ctx.meta['width']
        num_tiles_u = ctx.meta['num_tiles_u']
        tile_size = ctx.meta['tile_size']
        chi_square_clip = ctx.meta['chi_square_clip']
        alpha_max = ctx.meta['alpha_max']
        alpha_cutoff = ctx.meta['alpha_cutoff']

        # 将 grad_out 重塑为一维展平像素形状以便于处理
        # grad_out_flat shape: (height * width, 3)
        grad_out_flat = grad_out.view(-1, 3)

        # 初始化原始输入的颜色梯度
        # grad_colors shape: (N, 3)
        grad_colors = torch.zeros_like(colors)

        # 初始化已排序高斯球的不透明度梯度
        # grad_opacity_sorted shape: (N_onscreen,)
        grad_opacity_sorted = torch.zeros_like(opacity_sorted)

        # 2. 重新进行切片循环计算梯度 (Redo Tiling Loop)
        for tile_id, start, end in zip(unique_tile_ids.tolist(), unique_starts.tolist(), unique_ends.tolist()):
            txi = tile_id % num_tiles_u
            tyi = tile_id // num_tiles_u

            # 计算当前 Tile 的像素边界
            x0 = txi * tile_size
            y0 = tyi * tile_size
            x1 = min((txi + 1) * tile_size, width)
            y1 = min((tyi + 1) * tile_size, height)

            # 生成 Tile 内的二维像素网格
            xs = torch.arange(x0, x1, dtype=pos.dtype, device=pos.device)
            ys = torch.arange(y0, y1, dtype=pos.dtype, device=pos.device)
            px, py = torch.meshgrid(xs, ys, indexing='xy')

            # px_u/px_v shape: (P,), P 为当前 Tile 内的像素总数 (P <= tile_size * tile_size)
            px_u = px.reshape(-1)
            px_v = py.reshape(-1)
            # pixel_idx_1D shape: (P,)
            pixel_idx_1D = (px_v * width + px_u).to(torch.int64)

            # 获取当前 Tile 对应的 grad_out
            # grad_out_tile shape: (P, 3)
            grad_out_tile = grad_out_flat[pixel_idx_1D]

            # 获取当前 Tile 包含的高斯范围
            # ids_tile shape: (N_tile,), N_tile = end - start, 为当前 Tile 内的高斯球数量
            ids_tile = gaussian_ids_sorted[start:end]

            # 提取对应高斯的属性
            # u_tile/v_tile/opacity_tile shape: (N_tile,)
            u_tile = u_sorted[ids_tile]
            v_tile = v_sorted[ids_tile]
            opacity_tile = opacity_sorted[ids_tile]
            # inv_cov_tile shape: (N_tile, 2, 2)
            inv_cov_tile = inv_cov[ids_tile]

            # 重新计算权重 w
            # du/dv shape: (N_tile, P)
            du = px_u.unsqueeze(0) - u_tile.unsqueeze(1)
            dv = px_v.unsqueeze(0) - v_tile.unsqueeze(1)

            # a11/a12/a22 shape: (N_tile, 1)
            a11 = inv_cov_tile[:, 0, 0].unsqueeze(1)
            a12 = inv_cov_tile[:, 0, 1].unsqueeze(1)
            a22 = inv_cov_tile[:, 1, 1].unsqueeze(1)

            # Q shape: (N_tile, P)
            Q = a11 * du**2 + 2.0 * a12 * du * dv + a22 * dv**2

            # inside/G shape: (N_tile, P)
            inside = Q <= chi_square_clip
            G = torch.exp(-0.5 * Q)
            G = torch.where(inside, G, 0.0)

            # alpha/ti/w shape: (N_tile, P)
            alpha = opacity_tile.unsqueeze(1) * G
            alpha = torch.clamp(alpha, max=alpha_max)
            alpha = torch.where(alpha >= alpha_cutoff, alpha, 0.0)

            ti = torch.cumprod(1.0 - alpha, dim=0)
            ti = torch.cat([
                torch.ones((1, alpha.shape[1]), device=alpha.device, dtype=alpha.dtype),
                ti[:-1]
            ], dim=0)

            w = alpha * ti

            # 3. 计算颜色梯度：链式法则 dl/dcolor = grad_out_tile * w
            # w.unsqueeze(-1) shape: (N_tile, P, 1)
            # grad_out_tile.unsqueeze(0) shape: (1, P, 3)
            # 两者相乘后在 P (dim=1) 维度求和，得到 tile_grad_colors shape: (N_tile, 3)
            tile_grad_colors = (w.unsqueeze(-1) * grad_out_tile.unsqueeze(0)).sum(dim=1)

            # 4. 使用 scatter_add_ 直接将梯度累加回原始输入的形状上
            # indices_onscreen[ids_tile] shape: (N_tile,)
            orig_ids_tile = indices_onscreen[ids_tile]
            grad_colors.scatter_add_(0, orig_ids_tile.unsqueeze(1).expand(-1, 3), tile_grad_colors)

            # 5. 计算不透明度梯度（中间状态，只计算到对 alpha 的导数）
            colors_tile = colors_sorted[ids_tile]  # (N_tile, 3)
            cw = colors_tile.unsqueeze(1) * w.unsqueeze(-1)  # (N_tile, P, 3)
            cw_flip = torch.flip(cw, dims=[0])  # (N_tile, P, 3)
            s_flip = torch.cumsum(cw_flip, dim=0)  # (N_tile, P, 3)
            s_shifted_flip = torch.cat([
                torch.zeros((1, s_flip.shape[1], s_flip.shape[2]), device=s_flip.device, dtype=s_flip.dtype),
                s_flip[:-1]
            ], dim=0)  # (N_tile, P, 3)
            s = torch.flip(s_shifted_flip, dims=[0])  # (N_tile, P, 3)

            denom = torch.clamp(1.0 - alpha, min=1e-8).unsqueeze(-1)  # (N_tile, P, 1)
            d_out_d_alpha = colors_tile.unsqueeze(1) * ti.unsqueeze(-1) - s / denom  # (N_tile, P, 3)

            tile_grad_alpha = (d_out_d_alpha * grad_out_tile.unsqueeze(0)).sum(dim=-1)  # (N_tile, P)

            mask_active = (alpha >= alpha_cutoff) & (alpha < alpha_max)  # (N_tile, P)
            tile_grad_opacity = (tile_grad_alpha * G * mask_active).sum(dim=1)  # (N_tile,)

            grad_opacity_sorted.scatter_add_(0, ids_tile, tile_grad_opacity)



        # 其他不需要计算梯度的输入参数设置为零/None
        # grad_pos shape: (N, 3)
        grad_pos = torch.zeros_like(pos)
        # 反向投影：将排序后高斯球的梯度映射回原始输入的形状
        grad_opacity_raw = torch.zeros_like(opacity_raw)
        grad_opacity_raw.index_add_(0, indices_onscreen, grad_opacity_sorted)
        # grad_sigma shape: (N, 3, 3)
        grad_sigma = torch.zeros_like(sigma)

        return (
            grad_pos,
            grad_colors,
            grad_opacity_raw,
            None,  # height
            None,  # width
            None,  # fx
            None,  # fy
            None,  # cx
            None,  # cy
            None,  # camera2world
            grad_sigma,
            None,  # near
            None,  # far
            None,  # pixelGuard
            None,  # tile_size
            None,  # min_conic
            None,  # chi_square_clip
            None,  # alpha_max
            None,  # alpha_cutoff
        )
