import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rasterizer_function import RasterizerFunction
from util import w2c_to_c2w, compute_3d_covariance, scale_intrinsics, load_cameras, build_gaussian_from_sfm, evaluate_sh, makeOptimizer

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

def compute_loss(pred, target):
    # 将形状从 (H, W, C) 转换为 (1, C, H, W) 以符合 TorchMetrics 的要求
    pred_trans = pred.permute(2, 0, 1).unsqueeze(0)
    target_trans = target.permute(2, 0, 1).unsqueeze(0)

    l1 = F.l1_loss(pred, target)
    ssim_val = ssim(pred_trans, target_trans)
    return 0.8 * l1 + 0.2 * (1.0 - ssim_val)

# 6. 设定优化变量与 PyTorch 参数
pos = torch.nn.Parameter(initial_pos.clone())
f_dc = torch.nn.Parameter(params["f_dc"].clone())
f_rest = torch.nn.Parameter(params["f_rest"].clone())
alpha_raw = torch.nn.Parameter(initial_alpha_raw.clone())
rot_raw = torch.nn.Parameter(initial_rot_raw.clone())
scale_raw = torch.nn.Parameter(initial_scale_raw.clone())

opt_params = {
    "pos": pos,
    "f_dc": f_dc,
    "f_rest": f_rest,
    "alpha_raw": alpha_raw,
    "scale_raw": scale_raw,
    "rot_raw": rot_raw
}
optimizer = makeOptimizer(opt_params)

# 7. 核心训练/优化循环
print("\n--- Start Joint Parameter Optimization Loop ---", flush=True)
import os
from tqdm import tqdm
num_iterations = int(os.environ.get("NUM_ITERATIONS", 7000))
loss_history = []
psnr_history = []

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

    # 动态评估当前视角下的高斯 RGB 颜色
    colors = evaluate_sh(f_dc, f_rest, pos, c2w, interleaved=False)

    # 渲染当前优化器下参数的图像
    pred_image = RasterizerFunction.apply(
        pos, colors, alpha_raw, H_c, W_c, fx_c, fy_c, cx_c, cy_c, c2w, sigma
    )

    # 计算 L1 + SSIM 损失
    loss = compute_loss(pred_image, target_image_c)

    # 零梯度、反向传播与优化器更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)

    # 计算并记录 PSNR (使用 torchmetrics)
    pred_trans = pred_image.detach().permute(2, 0, 1).unsqueeze(0)
    target_trans = target_image_c.permute(2, 0, 1).unsqueeze(0)
    psnr_val = psnr(pred_trans, target_trans, data_range=1.0).item()
    psnr_history.append(psnr_val)

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
    final_np = (final_image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(final_np).save("verify_optimized.png")
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
