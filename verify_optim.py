import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rasterizer_function import RasterizerFunction
from util import w2c_to_c2w, compute_3d_covariance, scale_intrinsics, load_cameras

# 1. 配置 GPU 设备与数据路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}", flush=True)

# 2. 从 datasets/out_colmap/bonsai/point_cloud.npy 加载点云进行初始化
print("Loading Bonsai COLMAP point cloud...", flush=True)
pc_path = "datasets/out_colmap/bonsai/point_cloud.npy"
pc_data = np.load(pc_path)

# 提取位置坐标 pos 和真正的初始化颜色 init_colors
pos = torch.from_numpy(pc_data[:, :3]).to(device).float()
init_colors = torch.from_numpy(pc_data[:, 3:6]).to(device).float() / 255.0

# 高斯点总数 N
N = pos.shape[0]

# 初始化不透明度 alpha_raw (设置为 logit(0.1) = -2.1972)
initial_alpha_raw = torch.full((N,), -2.1972, device=device)
alpha_raw = torch.nn.Parameter(initial_alpha_raw.clone())

# 初始化旋转 rot_raw (设置为无旋转 [1.0, 0.0, 0.0, 0.0])
initial_rot_raw = torch.zeros((N, 4), device=device)
initial_rot_raw[:, 0] = 1.0
rot_raw = torch.nn.Parameter(initial_rot_raw)

# 初始化缩放 scale_raw (设置为各向异性缩放，避免旋转梯度为 0)
initial_scale_raw = torch.log(torch.tensor([0.01, 0.02, 0.03], device=device).unsqueeze(0).repeat(N, 1))
scale_raw = torch.nn.Parameter(initial_scale_raw)

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

# 选择第一个视角及其对应的世界坐标系外参 c2w 与图像路径
c2w = c2ws[0]
img_path = image_paths[0]

# 4 倍缩放
downscale = 4
w_half = int(width / downscale)
h_half = int(height / downscale)
fx_h, fy_h, cx_h, cy_h = scale_intrinsics(w_half, h_half, width, height, fx, fy, cx, cy)

print(f"Original Resolution: {width}x{height} | Target Resolution: {w_half}x{h_half}", flush=True)

# 4. 设置优化目标图像 (读取对应的 GT 图像)
# target_image = torch.zeros((h_half, w_half, 3), device=device)
# print("Target image set to all-zero black map (no forward rendering needed).", flush=True)

print(f"Loading GT image from {img_path}...", flush=True)
real_img = Image.open(img_path)
real_img_resized = real_img.resize((w_half, h_half), Image.Resampling.LANCZOS)
target_image = torch.from_numpy(np.array(real_img_resized)).to(device).float() / 255.0

# 5. 设定优化变量
# 方式 A：使用全黑初始化 (配合真实 GT 图像优化时使用)
# initial_colors = torch.zeros_like(init_colors)
# 方式 B：使用随机颜色初始化 (配合黑图 target_image 验证梯度时使用)
# initial_colors = torch.rand_like(init_colors)
# 方式 C：使用点云自带的真实颜色初始化
initial_colors = init_colors.clone()

colors = torch.nn.Parameter(initial_colors)

# 创建 Adam 优化器 (对颜色、不透明度、缩放和旋转参数进行优化)
optimizer = torch.optim.Adam([
    {"params": colors, "lr": 0.02},
    {"params": alpha_raw, "lr": 0.05},
    {"params": scale_raw, "lr": 0.005},
    {"params": rot_raw, "lr": 0.002}
])

# 6. 核心训练/优化循环
print("\n--- Start Real-Data Joint Parameter Optimization Loop ---", flush=True)
loss_history = []

# 定义中途评估感兴趣的 Epoch 阶段 (共 9 个阶段)
interested_epochs = [1, 7, 14, 21, 28, 35, 42, 49, 60]
stage_images = {}

for epoch in range(60):
    # 动态计算当前参数下的 3D 协方差矩阵 (sigma)
    sigma = compute_3d_covariance(scale_raw, rot_raw)

    # 渲染当前优化器下参数的图像
    pred_image = RasterizerFunction.apply(
        pos, colors, alpha_raw, h_half, w_half, fx_h, fy_h, cx_h, cy_h, c2w, sigma
    )

    # 记录指定阶段的渲染图像，用于后续 3x3 评估对比
    if (epoch + 1) in interested_epochs:
        stage_images[epoch + 1] = pred_image.detach().clamp(0.0, 1.0).cpu().numpy()

    # 仅使用 L1 损失函数进行监督
    loss = F.l1_loss(pred_image, target_image)

    # 零梯度、反向传播与优化器更新
    optimizer.zero_grad()
    loss.backward()

    # 验证不透明度与协方差参数梯度
    if epoch == 0:
        alpha_grad_norm = alpha_raw.grad.norm().item()
        alpha_grad_mean = alpha_raw.grad.abs().mean().item()
        print(f"--- Gradient Verification (Step 1) ---", flush=True)
        print(f"alpha_raw gradient norm: {alpha_grad_norm:.6f}", flush=True)
        print(f"alpha_raw gradient mean: {alpha_grad_mean:.6f}", flush=True)

        scale_grad_norm = scale_raw.grad.norm().item() if scale_raw.grad is not None else 0.0
        scale_grad_mean = scale_raw.grad.abs().mean().item() if scale_raw.grad is not None else 0.0
        print(f"scale_raw gradient norm: {scale_grad_norm:.6f}", flush=True)
        print(f"scale_raw gradient mean: {scale_grad_mean:.6f}", flush=True)

        rot_grad_norm = rot_raw.grad.norm().item() if rot_raw.grad is not None else 0.0
        rot_grad_mean = rot_raw.grad.abs().mean().item() if rot_raw.grad is not None else 0.0
        print(f"rot_raw gradient norm:   {rot_grad_norm:.6f}", flush=True)
        print(f"rot_raw gradient mean:   {rot_grad_mean:.6f}", flush=True)
        print(f"--------------------------------------", flush=True)

    optimizer.step()

    loss_val = loss.item()
    loss_history.append(loss_val)
    print(f"Step {epoch+1:03d} | L1 Loss: {loss_val:.6f}", flush=True)

# 7. 渲染并保存最终优化后的图像以供目视对比
with torch.no_grad():
    sigma = compute_3d_covariance(scale_raw, rot_raw)
    final_image = RasterizerFunction.apply(
        pos, colors, alpha_raw, h_half, w_half, fx_h, fy_h, cx_h, cy_h, c2w, sigma
    )
    final_np = (final_image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(final_np).save("verify_optimized.png")
    print("Saved optimized result image to 'verify_optimized.png'", flush=True)

# 8. 打印与初始化的平均颜色和不透明度绝对差
color_diff = torch.mean(torch.abs(colors.data - init_colors)).item()
print(f"\nFinal optimized color mean absolute difference from initialization: {color_diff:.6f}", flush=True)
alpha_diff = torch.mean(torch.abs(alpha_raw.data - initial_alpha_raw)).item()
print(f"Final optimized alpha_raw mean absolute difference from initialization: {alpha_diff:.6f}", flush=True)
scale_diff = torch.mean(torch.abs(scale_raw.data - initial_scale_raw)).item()
print(f"Final optimized scale_raw mean absolute difference from initialization: {scale_diff:.6f}", flush=True)
rot_diff = torch.mean(torch.abs(rot_raw.data - initial_rot_raw)).item()
print(f"Final optimized rot_raw mean absolute difference from initialization:   {rot_diff:.6f}", flush=True)

# 9. 绘制并保存 Loss 曲线图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='b', label='L1 Loss')
plt.title("L1 Loss Convergence Curve (lr=0.02)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("verify_loss.png")
print("Saved convergence curve plot to 'verify_loss.png'", flush=True)

# 10. 绘制 9 个阶段的前向渲染图像演变图 (3x3 网格图)
fig, axes = plt.subplots(3, 3, figsize=(15, 10), facecolor='white')
axes = axes.flatten()
for idx, ep in enumerate(interested_epochs):
    img = stage_images.get(ep)
    if img is not None:
        axes[idx].imshow(img)
        axes[idx].set_title(f"Step {ep}", fontsize=12, fontweight='bold')
    axes[idx].axis('off')

plt.suptitle("3DGS Color Optimization: Evolution of Forward Rendering", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("verify_stages.png", dpi=150)
print("Saved stage evolution plot to 'verify_stages.png'", flush=True)
