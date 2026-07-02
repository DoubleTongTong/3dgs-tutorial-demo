import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rasterizer_function import RasterizerFunction
from util import w2c_to_c2w, compute_3d_covariance, scale_intrinsics, load_cameras, build_gaussian_from_sfm, evaluate_sh, makeOptimizer, clone_gaussians, split_gaussians, prune_gaussians, tensor_to_pil, validate


# 1. 配置 GPU 设备与数据路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}", flush=True)

# 2. 从 datasets/out_colmap/bonsai/point_cloud.npy 加载点云并初始化高斯参数
print("Loading Bonsai COLMAP point cloud and building Gaussians...", flush=True)
pc_path = "datasets/out_colmap/bonsai/point_cloud.npy"
params = build_gaussian_from_sfm(pc_path, device=device, dtype=torch.float32, alpha_init=0.1)

initial_pos = params["pos"].clone()
initial_alpha_raw = params["alpha_raw"].clone()
initial_rot_raw = params["rot_raw"].clone()
initial_scale_raw = params["scale_raw"].clone()

SH_C0 = 0.28209479177387814
init_colors = torch.sigmoid(params["f_dc"] * SH_C0)
initial_colors = init_colors.clone()

# 高斯点总数 N
N = initial_pos.shape[0]

# 3. 加载相机参数并进行 4 倍降采样
print("Loading and downscaling camera intrinsics...", flush=True)
cam_meta = np.load("datasets/out_colmap/bonsai/cam_meta.npy", allow_pickle=True).item()

# 使用 load_cameras 加载真视角对应的图像路径与外参
c2ws, image_paths = load_cameras(
    cameras_path="datasets/out_colmap/bonsai/cameras.npy",
    images_root="datasets/360_v2/bonsai/images_2",
    device=device
)

# 自动计算场景比例 (scene_scale)
c2ws_stacked = torch.stack(c2ws)
camera_positions = c2ws_stacked[:, :3, 3]
scene_center = torch.mean(camera_positions, dim=0)
distances = torch.norm(camera_positions - scene_center, dim=-1)
scene_scale = torch.max(distances).item() * 1.1

height = cam_meta['height']
width = cam_meta['width']
fx, fy = cam_meta['fx'], cam_meta['fy']
cx, cy = width / 2, height / 2

# 4. 加载目标图像并做 Train/Test 划分 (不再固定降采样)
target_images = []
print("Loading all target images...", flush=True)
for img_path in image_paths:
    real_img = Image.open(img_path)
    target_image = torch.from_numpy(np.array(real_img)).to(device).float() / 255.0
    target_images.append(target_image)

h_base, w_base = target_images[0].shape[0], target_images[0].shape[1]
fx_base, fy_base, cx_base, cy_base = scale_intrinsics(w_base, h_base, width, height, fx, fy, cx, cy)

training_indices = []
testing_indices = []
for i in range(len(c2ws)):
    if i % 8 == 0:
        testing_indices.append(i)
    else:
        training_indices.append(i)
print(f"Dataset split: {len(training_indices)} train images, {len(testing_indices)} test images.", flush=True)

# 5. 使用 TorchMetrics 定义 SSIM 与混合损失函数
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr

def get_sh_bound_mask(bound, device="cuda"):
    if bound == 0:
        num_terms = 0
    elif bound == 1:
        num_terms = 3
    elif bound == 2:
        num_terms = 8
    else:
        num_terms = 15
    mask_15 = torch.zeros(15, device=device)
    if num_terms > 0:
        mask_15[:num_terms] = 1.0
    return torch.cat([mask_15, mask_15, mask_15], dim=0)

def apply_sh_masking(f_rest, iteration):
    bound = min(iteration // 1000, 3)
    mask = get_sh_bound_mask(bound, device=f_rest.device)
    return f_rest * mask

def compute_loss(pred, target):
    # 将形状从 (H, W, C) 转换为 (1, C, H, W) 以符合 TorchMetrics 的要求
    pred_trans = pred.permute(2, 0, 1).unsqueeze(0)
    target_trans = target.permute(2, 0, 1).unsqueeze(0)

    l1 = F.l1_loss(pred, target)
    ssim_val = ssim(pred_trans, target_trans, data_range=1.0)
    return 0.8 * l1 + 0.2 * (1.0 - ssim_val)

# 6. 设定优化变量与 PyTorch 参数
pos = torch.nn.Parameter(initial_pos.clone())
f_dc = torch.nn.Parameter(params["f_dc"].clone())
f_rest = torch.nn.Parameter(params["f_rest"].clone())
alpha_raw = torch.nn.Parameter(initial_alpha_raw.clone())
rot_raw = torch.nn.Parameter(initial_rot_raw.clone())
scale_raw = torch.nn.Parameter(initial_scale_raw.clone())

# 初始化一维视空间位置梯度累加器与可见性分母计数器
pos.sum_g_view = torch.zeros(pos.shape[0], device=device)  # (N,)
pos.denom = torch.zeros(pos.shape[0], device=device)  # (N,)

opt_params = {
    "pos": pos,
    "f_dc": f_dc,
    "f_rest": f_rest,
    "alpha_raw": alpha_raw,
    "scale_raw": scale_raw,
    "rot_raw": rot_raw
}
optimizer = makeOptimizer(opt_params)

# 自适应密度控制（Densification）超参数
tau_pos = 0.0002          # 触发分裂/克隆的位置梯度阈值 (2e-4)
tau_scale = 0.01 * scene_scale  # 区分克隆与分裂的高斯尺寸缩放阈值
epsilon_alpha = 0.005      # 剪枝时的低透明度截断阈值
toe_size_3d = 0.1 * scene_scale  # 大尺寸高斯剪枝阈值


# 7. 核心训练/优化循环
print("\n--- Start Joint Parameter Optimization Loop ---", flush=True)
import os
from tqdm import tqdm
num_iterations = int(os.environ.get("NUM_ITERATIONS", 7000))
loss_history = []
psnr_history = []
gaussian_count_history = []
validation_psnr = []
validation_ssim = []
log_dir = "training_out"

for iteration in tqdm(range(num_iterations)):
    # 随机选择一个训练图像视角
    idx = np.random.randint(0, len(training_indices))
    view_index = training_indices[idx]

    c2w = c2ws[view_index]
    target_image = target_images[view_index]

    # 动态分辨率缩放 (Warm-up)
    if iteration < 250:
        S = 0.25
    elif iteration < 500:
        S = 0.5
    else:
        S = 1.0

    H_c = int(S * h_base)
    W_c = int(S * w_base)

    if S < 1.0:
        fx_c, fy_c, cx_c, cy_c = scale_intrinsics(W_c, H_c, w_base, h_base, fx_base, fy_base, cx_base, cy_base)
        # 缩放目标图像
        target_image_trans = target_image.permute(2, 0, 1).unsqueeze(0)
        target_image_resized = F.interpolate(target_image_trans, size=(H_c, W_c), mode='bilinear', align_corners=False)
        target_image_c = target_image_resized.squeeze(0).permute(1, 2, 0)
    else:
        fx_c, fy_c, cx_c, cy_c = fx_base, fy_base, cx_base, cy_base
        target_image_c = target_image

    # 动态计算当前参数下的 3D 协方差矩阵 (sigma)
    sigma = compute_3d_covariance(scale_raw, rot_raw)

    # 动态评估当前视角下的高斯 RGB 颜色并应用球谐函数分阶激活遮罩
    f_rest_effective = apply_sh_masking(f_rest, iteration)
    colors = evaluate_sh(f_dc, f_rest_effective, pos, c2w, interleaved=False)

    # 渲染当前优化器下参数的图像
    pred_image = RasterizerFunction.apply(
        pos, colors, alpha_raw, H_c, W_c, fx_c, fy_c, cx_c, cy_c, c2w, sigma
    )

    # 计算 L1 + SSIM 损失
    loss = compute_loss(pred_image, target_image_c)

    # 零梯度、反向传播与优化器更新
    optimizer.zero_grad()
    loss.backward()

    # 只在自适应密度控制的活跃区间内（500 到 3000 步之间）累加 2D 视空间位置梯度幅值与可见性分母
    if iteration > 500 and iteration <= 3000:
        with torch.no_grad():
            # RasterizerFunction.gView: (N, 2)
            # gView_norm: (N,) (计算 2D 梯度向量的 L2 范数/模长)
            gView_norm = torch.norm(RasterizerFunction.gView, dim=-1)
            # pos.sum_g_view: (N,)
            pos.sum_g_view += gView_norm
            # RasterizerFunction.visible_mask: (N,)
            # pos.denom: (N,) (可见性分母计数器累加)
            pos.denom += RasterizerFunction.visible_mask

    optimizer.step()

    # 验证逻辑 (每 1000 步，或者当调试环境变量 DEBUG_VALIDATE == 1 时在每步执行)
    do_validate = (iteration % 1000 == 0) or (os.environ.get("DEBUG_VALIDATE") == "1")
    if do_validate:
        # 在调试模式下只评估第一个测试视角以极大加快运行速度
        val_indices = testing_indices[:1] if os.environ.get("DEBUG_VALIDATE") == "1" else testing_indices
        iter_log_dir = os.path.join(log_dir, f"iteration_{iteration}")
        val_psnr, val_ssim = validate(
            pos, f_dc, f_rest, scale_raw, rot_raw, alpha_raw,
            c2ws, target_images, val_indices,
            h_base, w_base, fx_base, fy_base, cx_base, cy_base,
            log_dir=iter_log_dir, iteration=iteration
        )
        validation_psnr.append(val_psnr)
        validation_ssim.append(val_ssim)
        tqdm.write(f"Validation at Iteration {iteration} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f} | Num Gaussians: {pos.shape[0]}")

        # 绘图逻辑
        os.makedirs(iter_log_dir, exist_ok=True)

        # 1. 验证 SSIM 趋势图
        if len(validation_ssim) > 0:
            plt.figure()
            plt.plot(validation_ssim, color='g')
            plt.grid(True)
            plt.title("Validation SSIM")
            plt.xlabel("Validation Event")
            plt.ylabel("SSIM")
            plt.savefig(os.path.join(iter_log_dir, "validation_ssim.png"))
            plt.close()

        # 2. 验证 PSNR 趋势图
        if len(validation_psnr) > 0:
            plt.figure()
            plt.plot(validation_psnr, color='r')
            plt.grid(True)
            plt.title("Validation PSNR")
            plt.xlabel("Validation Event")
            plt.ylabel("PSNR (dB)")
            plt.savefig(os.path.join(iter_log_dir, "validation_psnr.png"))
            plt.close()

        # 3. 训练 Loss 趋势图
        if len(loss_history) > 0:
            plt.figure()
            plt.plot(loss_history, color='b')
            plt.grid(True)
            plt.title("Training Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(iter_log_dir, "training_loss.png"))
            plt.close()

        # 4. 训练 PSNR 趋势图
        if len(psnr_history) > 0:
            plt.figure()
            plt.plot(psnr_history, color='m')
            plt.grid(True)
            plt.title("Training PSNR")
            plt.xlabel("Iteration")
            plt.ylabel("PSNR (dB)")
            plt.savefig(os.path.join(iter_log_dir, "training_psnr.png"))
            plt.close()

        # 5. 高斯点数变化趋势图
        if len(gaussian_count_history) > 0:
            plt.figure()
            plt.plot(gaussian_count_history, color='c')
            plt.grid(True)
            plt.title("Gaussian Count")
            plt.xlabel("Iteration")
            plt.ylabel("Number of Gaussians")
            plt.savefig(os.path.join(iter_log_dir, "gaussian_count.png"))
            plt.close()


    # 自适应密度控制与剪枝 (Densification & Pruning)
    # 起始步：500 步，每 100 步触发一次。克隆与分裂在 3000 步前执行，体积大剪枝在 3000 步后执行。
    if iteration > 500 and iteration % 100 == 0:
        with torch.no_grad():
            if iteration <= 3000:
                # 1. 筛选高梯度高斯点 (通过累计梯度除以可见次数得到真正的平均梯度)
                # pos.sum_g_view: (N,), pos.denom: (N,) -> is_high_grad: (N,)
                is_high_grad = (pos.sum_g_view / torch.clamp(pos.denom, min=1.0)) > tau_pos

                # 2. 计算高斯缩放
                scales = torch.exp(scale_raw)  # (N, 3)
                max_scales = torch.max(scales, dim=1).values
                is_big = max_scales > tau_scale
                is_small = ~is_big

                mask_clone = is_high_grad & is_small
                mask_split = is_high_grad & is_big

                # 3. 执行克隆
                if mask_clone.any():
                    opt_params, optimizer = clone_gaussians(mask_clone, opt_params, optimizer)

                # 4. 执行分裂
                if mask_split.any():
                    # 如果由于克隆导致高斯点数量增加，对 mask_split 在末尾用 False 填充以对齐当前维度
                    current_N = opt_params["pos"].shape[0]
                    if current_N > mask_split.shape[0]:
                        pad_len = current_N - mask_split.shape[0]
                        pad_tensor = torch.zeros(pad_len, dtype=torch.bool, device=mask_split.device)
                        mask_split = torch.cat([mask_split, pad_tensor], dim=0)
                    opt_params, optimizer = split_gaussians(mask_split, opt_params, optimizer)

            # 5. 执行剪枝（剔除透明高斯点）
            # 重新读取当前最新的 alpha_raw 形状
            alpha_raw_latest = opt_params["alpha_raw"]
            mask_prune = torch.sigmoid(alpha_raw_latest) < epsilon_alpha

            # 6. 在 3000 步之后，额外剔除在物理空间中体积过大的高斯点
            if iteration > 3000:
                scale_raw_latest = opt_params["scale_raw"]
                max_scales = torch.exp(scale_raw_latest).max(dim=1).values
                too_big = max_scales > toe_size_3d
                mask_prune = mask_prune | too_big

            # 为了防止剪掉所有的高斯，至少保留一个
            if mask_prune.all():
                mask_prune[0] = False

            if mask_prune.any():
                opt_params, optimizer = prune_gaussians(mask_prune, opt_params, optimizer)

            # 同步局部变量以配合后续循环迭代
            pos = opt_params["pos"]
            f_dc = opt_params["f_dc"]
            f_rest = opt_params["f_rest"]
            alpha_raw = opt_params["alpha_raw"]
            scale_raw = opt_params["scale_raw"]
            rot_raw = opt_params["rot_raw"]

            # 重置梯度累加与分母，以便为下 100 步重新累加
            pos.sum_g_view = torch.zeros(pos.shape[0], device=device)  # (N,)
            pos.denom = torch.zeros(pos.shape[0], device=device)  # (N,)

    loss_val = loss.item()
    loss_history.append(loss_val)

    # 计算并记录 PSNR (使用 torchmetrics)
    pred_trans = pred_image.detach().permute(2, 0, 1).unsqueeze(0)
    target_trans = target_image_c.permute(2, 0, 1).unsqueeze(0)
    psnr_val = psnr(pred_trans, target_trans, data_range=1.0).item()
    psnr_history.append(psnr_val)

    # 记录高斯点数量
    gaussian_count_history.append(pos.shape[0])

    if (iteration + 1) % 500 == 0 or iteration == 0:
        print(f"Iteration {iteration+1:04d} | Loss: {loss_val:.6f} | PSNR: {psnr_val:.4f}", flush=True)

# 8. 渲染并保存最终优化后的图像 (从测试集选择一个视角以验证效果)
with torch.no_grad():
    test_idx = testing_indices[0]
    test_c2w = c2ws[test_idx]
    sigma = compute_3d_covariance(scale_raw, rot_raw)
    final_colors = evaluate_sh(f_dc, f_rest, pos, test_c2w, interleaved=False)
    final_image = RasterizerFunction.apply(
        pos, final_colors, alpha_raw, h_base, w_base, fx_base, fy_base, cx_base, cy_base, test_c2w, sigma
    )
    tensor_to_pil(final_image).save("verify_optimized.png")
    print(f"Saved optimized test view rendering to 'verify_optimized.png'", flush=True)

# 9. 绘制并保存 Loss 与 PSNR 曲线图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss 曲线
ax1.plot(range(1, len(loss_history) + 1), loss_history, color='b', label='Joint Loss')
ax1.set_title("Joint Loss Convergence Curve (L1 + SSIM)")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()

# PSNR 曲线
ax2.plot(range(1, len(psnr_history) + 1), psnr_history, color='r', label='PSNR')
ax2.set_title("PSNR Convergence Curve")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("PSNR (dB)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("verify_loss.png")
print("Saved convergence curves plot to 'verify_loss.png'", flush=True)

# 10. 绘制并保存高斯点数量变化曲线图
fig_gc, ax_gc = plt.subplots(figsize=(8, 5))
ax_gc.plot(range(1, len(gaussian_count_history) + 1), gaussian_count_history, color='g', label='Gaussian Count')
ax_gc.set_title("Gaussian Count Convergence Curve")
ax_gc.set_xlabel("Iteration")
ax_gc.set_ylabel("Number of Gaussians")
ax_gc.grid(True)
ax_gc.legend()
plt.tight_layout()
plt.savefig("verify_gaussian_count.png")
print("Saved Gaussian count convergence curve to 'verify_gaussian_count.png'", flush=True)
