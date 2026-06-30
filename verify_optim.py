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

# 4. 加载与缩放所有目标图像并做 Train/Test 划分
downscale = 4
w_half = int(width / downscale)
h_half = int(height / downscale)
fx_h, fy_h, cx_h, cy_h = scale_intrinsics(w_half, h_half, width, height, fx, fy, cx, cy)

target_images = []
print("Loading and resizing all target images...", flush=True)
for img_path in image_paths:
    real_img = Image.open(img_path)
    real_img_resized = real_img.resize((w_half, h_half), Image.Resampling.LANCZOS)
    target_image = torch.from_numpy(np.array(real_img_resized)).to(device).float() / 255.0
    target_images.append(target_image)

training_indices = []
testing_indices = []
for i in range(len(c2ws)):
    if i % 8 == 0:
        testing_indices.append(i)
    else:
        training_indices.append(i)
print(f"Dataset split: {len(training_indices)} train images, {len(testing_indices)} test images.", flush=True)

# 5. 定义 SSIM 与混合损失函数
def ssim(img1, img2, window_size=11):
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)

    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / 4.5) for x in range(window_size)])
    kernel1d = (gauss / gauss.sum()).unsqueeze(1)
    kernel2d = kernel1d.mm(kernel1d.t()).float().unsqueeze(0).unsqueeze(0)

    channels = img1.size(1)
    window = kernel2d.expand(channels, 1, window_size, window_size).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_loss(pred, target):
    l1 = F.l1_loss(pred, target)
    ssim_val = ssim(pred, target)
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

for iteration in tqdm(range(num_iterations)):
    # 随机选择一个训练图像视角
    idx = np.random.randint(0, len(training_indices))
    view_index = training_indices[idx]

    c2w = c2ws[view_index]
    target_image = target_images[view_index]

    # 动态计算当前参数下的 3D 协方差矩阵 (sigma)
    sigma = compute_3d_covariance(scale_raw, rot_raw)

    # 动态评估当前视角下的高斯 RGB 颜色
    colors = evaluate_sh(f_dc, f_rest, pos, c2w, interleaved=False)

    # 渲染当前优化器下参数的图像
    pred_image = RasterizerFunction.apply(
        pos, colors, alpha_raw, h_half, w_half, fx_h, fy_h, cx_h, cy_h, c2w, sigma
    )

    # 计算 L1 + SSIM 损失
    loss = compute_loss(pred_image, target_image)

    # 零梯度、反向传播与优化器更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)

    if (iteration + 1) % 500 == 0 or iteration == 0:
        print(f"Iteration {iteration+1:04d} | Loss: {loss_val:.6f}", flush=True)

# 8. 渲染并保存最终优化后的图像 (从测试集选择一个视角以验证效果)
with torch.no_grad():
    test_idx = testing_indices[0]
    test_c2w = c2ws[test_idx]
    sigma = compute_3d_covariance(scale_raw, rot_raw)
    final_colors = evaluate_sh(f_dc, f_rest, pos, test_c2w, interleaved=False)
    final_image = RasterizerFunction.apply(
        pos, final_colors, alpha_raw, h_half, w_half, fx_h, fy_h, cx_h, cy_h, test_c2w, sigma
    )
    final_np = (final_image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(final_np).save("verify_optimized.png")
    print(f"Saved optimized test view rendering to 'verify_optimized.png'", flush=True)

# 9. 绘制并保存 Loss 曲线图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, color='b', label='Joint Loss')
plt.title("Joint Loss Convergence Curve (L1 + SSIM)")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("verify_loss.png")
print("Saved convergence curve plot to 'verify_loss.png'", flush=True)
