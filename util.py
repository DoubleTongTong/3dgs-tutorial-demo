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

def compute_3d_covariance(scale_raw, rot_raw):
    """
    从缩放和旋转参数计算 3D 协方差矩阵 Sigma。
    rot_raw: 四元数参数，Shape (..., 4)，格式为 [w, x, y, z]
    scale_raw: 缩放参数，Shape (..., 3)
    """
    # rot_raw 的四元数格式为 [w, x, y, z]，需要转换为 quat_to_rotmat 期望的 [x, y, z, w]
    rot_xyzw = torch.cat([rot_raw[..., 1:], rot_raw[..., :1]], dim=-1)
    R = quat_to_rotmat(rot_xyzw)
    S = torch.exp(scale_raw)
    S2 = torch.diag_embed(S ** 2)
    sigma = R @ S2 @ R.transpose(-1, -2)
    return sigma

# -------------------------
# 球谐函数 (Spherical Harmonics) 常量及评估函数
# -------------------------

# 0阶系数 (Degree 0)
SH_C0 = 0.28209479177387814

# 1阶系数 (Degree 1) - 对应于 x, y, z 分量
SH_C1_y = 0.4886025119029199
SH_C1_z = 0.4886025119029199
SH_C1_x = 0.4886025119029199

# 2阶系数 (Degree 2) - 对应于多项式基底
SH_C2_xy = 1.0925484305920792
SH_C2_yz = 1.0925484305920792
SH_C2_zz = 0.31539156525252005
SH_C2_xz = 1.0925484305920792
SH_C2_xx_yy = 0.5462742152960396

# 3阶系数 (Degree 3) - 对应于更高次项
SH_C3_y_3x2_y2 = 0.5900435899266435
SH_C3_xyz = 2.890611442640554
SH_C3_y_zz_x2_y2 = 0.4570457994644658
SH_C3_zz_x2_y2 = 0.3731763325901154
SH_C3_x_zz_x2_y2 = 0.4570457994644658
SH_C3_z_x2_y2 = 1.445305721320277
SH_C3_x_x2_3y2 = 0.5900435899266435


def evaluate_sh(f_dc, f_rest, points, camera_to_world, interleaved=True):
    """
    计算基于视角的球谐颜色
    """
    # 1. 提取相机在世界坐标系中的位置 (最后一列前三个元素)
    camera_center = camera_to_world[:3, 3]

    # 2. 计算并归一化视角方向 (点到相机的向量)
    view_dir = points - camera_center
    view_dir = view_dir / (torch.norm(view_dir, dim=-1, keepdim=True) + 1e-12)

    # 3. 提取 x, y, z 分量
    x, y, z = view_dir[:, 0], view_dir[:, 1], view_dir[:, 2]

    # 4. 预计算高阶项乘积以提升效率
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # 5. 计算 16 个球谐基函数 (Y0 至 Y15)
    Y0 = torch.full_like(x, SH_C0)

    # 1阶
    Y1 = -SH_C1_y * y
    Y2 = SH_C1_z * z
    Y3 = -SH_C1_x * x

    # 2阶
    Y4 = SH_C2_xy * xy
    Y5 = -SH_C2_yz * yz
    Y6 = SH_C2_zz * (3.0 * zz - 1.0)
    Y7 = -SH_C2_xz * xz
    Y8 = SH_C2_xx_yy * (xx - yy)

    # 3阶
    Y9 = -SH_C3_y_3x2_y2 * y * (3.0 * xx - yy)
    Y10 = SH_C3_xyz * xy * z
    Y11 = -SH_C3_y_zz_x2_y2 * y * (4.0 * zz - xx - yy)
    Y12 = SH_C3_zz_x2_y2 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    Y13 = -SH_C3_x_zz_x2_y2 * x * (4.0 * zz - xx - yy)
    Y14 = SH_C3_z_x2_y2 * z * (xx - yy)
    Y15 = -SH_C3_x_x2_3y2 * x * (xx - 3.0 * yy)

    # 6. 将 Y 堆叠为 (N, 16)
    Y = torch.stack([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15], dim=-1)

    # 7. 重组系数矩阵为 (N, 16, 3)
    N = points.shape[0]
    sh = torch.empty((N, 16, 3), dtype=points.dtype, device=points.device)

    # 填充 0 阶系数 (f_dc)
    sh[:, 0, :] = f_dc

    # 填充 1-3 阶系数 (f_rest)
    if interleaved:
        sh[:, 1:, :] = f_rest.reshape(-1, 15, 3)
    else:
        sh[:, 1:, 0] = f_rest[:, :15]
        sh[:, 1:, 1] = f_rest[:, 15:30]
        sh[:, 1:, 2] = f_rest[:, 30:]

    # 8. 相乘求和并进行 Sigmoid 激活得到最终 RGB 颜色
    raw_rgb = torch.sum(sh * Y.unsqueeze(-1), dim=1)
    # return torch.clamp(raw_rgb + 0.5, min=0.0, max=1.0)
    return torch.sigmoid(raw_rgb)


def build_gaussian_from_sfm(data_path, device='cpu', dtype=torch.float32, alpha_init=0.05):
    """
    根据 SFM 点云数据构建 3D 高斯参数。
    支持传入文件路径 (.npy) 或已加载的张量/数组。
    """
    from pytorch3d.ops import knn_points

    pc_data = np.load(data_path)
    pc_data = torch.from_numpy(pc_data)

    # 提取位置与颜色，并移动到指定设备
    pos = pc_data[:, :3].to(device=device, dtype=dtype)
    colors = pc_data[:, 3:6].to(device=device, dtype=dtype)
    if colors.max() > 1.0:
        colors = colors / 255.0

    N = pos.shape[0]

    # 1. 0阶球谐系数 f_dc: f_dc = torch.logit(colors) / SH_C0
    # 为数值稳定，在 logit 之前对 colors 夹断保护
    colors_clamped = torch.clamp(colors, min=1e-4, max=1.0 - 1e-4)
    f_dc = torch.logit(colors_clamped) / SH_C0

    # 2. 1-3阶球谐系数 f_rest: 形状为 (N, 45)，初始化为全零
    f_rest = torch.zeros((N, 45), device=device, dtype=dtype)

    # 3. 旋转四元数 rot_raw: 形状为 (N, 4)，初始化为 [1, 0, 0, 0] 格式 (w, x, y, z)
    rot_raw = torch.zeros((N, 4), device=device, dtype=dtype)
    rot_raw[:, 0] = 1.0

    # 4. 缩放参数 scale_raw: 形状为 (N, 3)，由与 3 个最近邻的平均距离的对数初始化
    pos_batch = pos.unsqueeze(0)  # (1, N, 3)
    # K=4，因为包含自身（距离为0），真正查询的是 3 个最近邻
    knn_res = knn_points(pos_batch, pos_batch, K=min(4, N))

    # 获取除自身（第一个通道）以外的 3 个最近邻的平方距离
    # knn_res.dists 形状为 (1, N, K)
    if N > 1:
        sq_dists = knn_res.dists[0, :, 1:]  # (N, K-1)
        mean_dists = torch.mean(torch.sqrt(sq_dists), dim=-1, keepdim=True)  # (N, 1)
    else:
        mean_dists = torch.ones((N, 1), device=device, dtype=dtype) * 1e-2

    scale_raw = torch.log(mean_dists.clamp(min=1e-6)).repeat(1, 3)

    # 5. 不透明度原始参数 alpha_raw: 形状为 (N,)，对真实不透明度做 logit 变换
    alpha_raw_val = torch.logit(torch.tensor(alpha_init, device=device, dtype=dtype)).item()
    alpha_raw = torch.full((N,), alpha_raw_val, device=device, dtype=dtype)

    return {
        "pos": pos,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "alpha_raw": alpha_raw,
        "rot_raw": rot_raw,
        "scale_raw": scale_raw
    }


def makeOptimizer(parameters, lr_pos=1.6e-4, lr_fdc=2.5e-3, lr_frest=1.25e-4, lr_opacity=0.05, lr_scale=0.005, lr_q=0.001):
    """
    Create an Adam optimizer for 3D Gaussian Splatting parameters with custom learning rates.
    """
    param_groups = [
        {"params": parameters["pos"], "lr": lr_pos, "name": "pos"},
        {"params": parameters["f_dc"], "lr": lr_fdc, "name": "f_dc"},
        {"params": parameters["f_rest"], "lr": lr_frest, "name": "f_rest"},
        {"params": parameters["alpha_raw"], "lr": lr_opacity, "name": "alpha_raw"},
        {"params": parameters["scale_raw"], "lr": lr_scale, "name": "scale_raw"},
        {"params": parameters["rot_raw"], "lr": lr_q, "name": "rot_raw"},
    ]
    return torch.optim.Adam(param_groups, betas=(0.9, 0.99), eps=1e-15)


def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor (H, W, 3) or (3, H, W) or (1, 3, H, W) to a PIL Image.
    """
    from PIL import Image
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # If shape is (C, H, W), permute to (H, W, C)
    if tensor.shape[0] == 3 or tensor.shape[0] == 1:
        tensor = tensor.permute(1, 2, 0)

    arr = (tensor.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def update_optimizer_state(optimizer, new_parameters, map_state_fn):
    """
    Recreate the optimizer with new parameters and transfer/map the states. (TODO)
    """
    pass


def clone_gaussians(mask_clone, parameters, optimizer):
    """
    Clone selected Gaussians.
    """
    if not mask_clone.any():
        return parameters, optimizer

    for name, param in parameters.items():
        old_val = param.detach()
        cloned_val = old_val[mask_clone]
        new_val = torch.cat([old_val, cloned_val], dim=0)
        parameters[name] = torch.nn.Parameter(new_val, requires_grad=True)

    # Naive optimizer rebuild
    new_optimizer = makeOptimizer(parameters)
    return parameters, new_optimizer


def split_gaussians(mask_split, parameters, optimizer, N=2):
    """
    Split selected Gaussians. (TODO)
    """
    return parameters, optimizer


def prune_gaussians(mask_prune, parameters, optimizer):
    """
    Prune selected Gaussians. (TODO)
    """
    return parameters, optimizer
