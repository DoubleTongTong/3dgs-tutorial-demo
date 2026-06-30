import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rasterizer_function import RasterizerFunction
from util import w2c_to_c2w, compute_3d_covariance, scale_intrinsics, load_cameras, build_gaussian_from_sfm, evaluate_sh

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

# ==================== 视角选择变量 ====================
# 您可以通过修改 cam_index 来选择不同的相机视角进行渲染和优化
# 可选范围为 0 到 len(c2ws) - 1
cam_index = 5
# ====================================================

# 选择指定的相机视角及其对应的世界坐标系外参 c2w 与图像路径
c2w = c2ws[cam_index]
img_path = image_paths[cam_index]

# 4 倍缩放
downscale = 4
w_half = int(width / downscale)
h_half = int(height / downscale)
fx_h, fy_h, cx_h, cy_h = scale_intrinsics(w_half, h_half, width, height, fx, fy, cx, cy)

print(f"Original Resolution: {width}x{height} | Target Resolution: {w_half}x{h_half}", flush=True)

# 4. 设置优化目标图像 (读取对应的 GT 图像)
print(f"Loading GT image from {img_path}...", flush=True)
real_img = Image.open(img_path)
real_img_resized = real_img.resize((w_half, h_half), Image.Resampling.LANCZOS)
target_image = torch.from_numpy(np.array(real_img_resized)).to(device).float() / 255.0

# 5. 设定优化变量与 PyTorch 参数 (统一在此处进行 Parameter 包裹)
# 5.1 位置 pos
pos = torch.nn.Parameter(initial_pos.clone())

# 5.2 颜色相关：将 f_dc 与 f_rest 分别作为参数进行优化，实现球谐系数优化
f_dc = torch.nn.Parameter(params["f_dc"].clone())
f_rest = torch.nn.Parameter(params["f_rest"].clone())

# 5.3 不透明度 alpha_raw
alpha_raw = torch.nn.Parameter(initial_alpha_raw.clone())

# 5.4 旋转 rot_raw
rot_raw = torch.nn.Parameter(initial_rot_raw.clone())

# 5.5 缩放 scale_raw
scale_raw = torch.nn.Parameter(initial_scale_raw.clone())

# 创建 Adam 优化器 (对球谐系数、位置、不透明度、缩放和旋转参数进行联合优化)
optimizer = torch.optim.Adam([
    {"params": f_dc, "lr": 0.02},
    {"params": f_rest, "lr": 0.02},
    {"params": pos, "lr": 0.002},
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

    # 动态评估当前视角下的高斯 RGB 颜色
    colors = evaluate_sh(f_dc, f_rest, pos, c2w, interleaved=False)

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

    # 验证位置、不透明度与协方差参数梯度
    if epoch == 0:
        pos_grad_norm = pos.grad.norm().item() if pos.grad is not None else 0.0
        pos_grad_mean = pos.grad.abs().mean().item() if pos.grad is not None else 0.0
        alpha_grad_norm = alpha_raw.grad.norm().item()
        alpha_grad_mean = alpha_raw.grad.abs().mean().item()
        f_dc_grad_norm = f_dc.grad.norm().item() if f_dc.grad is not None else 0.0
        f_dc_grad_mean = f_dc.grad.abs().mean().item() if f_dc.grad is not None else 0.0
        f_rest_grad_norm = f_rest.grad.norm().item() if f_rest.grad is not None else 0.0
        f_rest_grad_mean = f_rest.grad.abs().mean().item() if f_rest.grad is not None else 0.0
        print(f"--- Gradient Verification (Step 1) ---", flush=True)
        print(f"pos gradient norm:       {pos_grad_norm:.6f}", flush=True)
        print(f"pos gradient mean:       {pos_grad_mean:.6f}", flush=True)
        print(f"alpha_raw gradient norm: {alpha_grad_norm:.6f}", flush=True)
        print(f"alpha_raw gradient mean: {alpha_grad_mean:.6f}", flush=True)
        print(f"f_dc gradient norm:      {f_dc_grad_norm:.6f}", flush=True)
        print(f"f_dc gradient mean:      {f_dc_grad_mean:.6f}", flush=True)
        print(f"f_rest gradient norm:    {f_rest_grad_norm:.6f}", flush=True)
        print(f"f_rest gradient mean:    {f_rest_grad_mean:.6f}", flush=True)

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
    final_colors = evaluate_sh(f_dc, f_rest, pos, c2w, interleaved=False)
    final_image = RasterizerFunction.apply(
        pos, final_colors, alpha_raw, h_half, w_half, fx_h, fy_h, cx_h, cy_h, c2w, sigma
    )
    final_np = (final_image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(final_np).save("verify_optimized.png")
    print("Saved optimized result image to 'verify_optimized.png'", flush=True)

# 8. 打印与初始化的平均坐标、颜色和不透明度绝对差
pos_diff = torch.mean(torch.abs(pos.data - initial_pos)).item()
print(f"\nFinal optimized pos mean absolute difference from initialization:       {pos_diff:.6f}", flush=True)
color_diff = torch.mean(torch.abs(final_colors - init_colors)).item()
print(f"Final optimized color mean absolute difference from initialization:     {color_diff:.6f}", flush=True)
f_dc_diff = torch.mean(torch.abs(f_dc.data - params["f_dc"])).item()
print(f"Final optimized f_dc mean absolute difference from initialization:       {f_dc_diff:.6f}", flush=True)
f_rest_diff = torch.mean(torch.abs(f_rest.data - params["f_rest"])).item()
print(f"Final optimized f_rest mean absolute difference from initialization:     {f_rest_diff:.6f}", flush=True)
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

plt.suptitle("3DGS Optimization: Evolution of Forward Rendering", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("verify_stages.png", dpi=150)
print("Saved stage evolution plot to 'verify_stages.png'", flush=True)
