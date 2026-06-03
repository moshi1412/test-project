import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.pcd_utils import fetchPly

# ===================== 你的路径 =====================
PLY_PATH    = "/mnt/ljy/street_crafter/data/waymo/000/lidar/background/000000.ply"
IMG_PATH    = "/mnt/ljy/street_crafter/data/waymo/000/images/000000_0.png"
DEPTH_NPY   = "/mnt/ljy/street_crafter/data/waymo/000/lidar/color_render/000000_0_depth.npy"
POSE_PATH   = "/mnt/ljy/street_crafter/data/waymo/000/ego_pose/000000.txt"
INT_PATH    = "/mnt/ljy/street_crafter/data/waymo/000/intrinsics/0.txt"
EXT_PATH    = "/mnt/ljy/street_crafter/data/waymo/000/extrinsics/0.txt"
# ======================================================

# --------------------- 1. 读取点云 ---------------------
ply = fetchPly(PLY_PATH)
xyz_vehicle = ply.points[ply.mask]
rgb = ply.colors[ply.mask]

# --------------------- 2. 坐标变换（完全对齐你工程） ---------------------
ego_pose = np.loadtxt(POSE_PATH).reshape(4,4)
ext = np.loadtxt(EXT_PATH).reshape(4,4)
intrinsic = np.loadtxt(INT_PATH)
fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]

# 车体坐标系 → 世界坐标系
xyz_homo = np.concatenate([xyz_vehicle, np.ones((len(xyz_vehicle), 1))], axis=1)
xyz_world = (xyz_homo @ ego_pose.T)[:, :3]

# 世界 → 相机坐标系
RT = np.linalg.inv(ego_pose @ ext)
xyz_cam = xyz_world @ RT[:3,:3].T + RT[:3, 3]

# --------------------- 3. 投影到图像（红单点） ---------------------
img = cv2.imread(IMG_PATH)
H, W = img.shape[:2]
Z = xyz_cam[:, 2]
U = (fx * xyz_cam[:, 0] / Z + cx).astype(np.int32)
V = (fy * xyz_cam[:, 1] / Z + cy).astype(np.int32)

# 过滤在图像范围内的点（避免画到外面）
mask = (U >= 0) & (U < W) & (V >= 0) & (V < H)
U_valid = U[mask]
V_valid = V[mask]

# ===================== Matplotlib 绘图 =====================
plt.figure(figsize=(W/100, H/100), dpi=100)  # 保持原图尺寸

# 画红色点（对应你原来的 (0,0,255)）
plt.scatter(
    U_valid, V_valid, 
    color=rgb[mask], 
    s=2        # 点大小，可调整
)

# 图像坐标系：Matplotlib 默认 y 向上，而图像是 y 向下，必须反转！
plt.gca().invert_yaxis()

# 设置范围和边框
plt.xlim(0, W)
plt.ylim(H, 0)
plt.axis('off')  # 去掉坐标轴

# 保存（透明背景 / 白色背景 都支持）
plt.savefig(
    "lidar_2d_red_matplotlib.png",
    bbox_inches='tight',
    pad_inches=0,
    dpi=100
)
plt.close()

# --------------------- 4. 深度图（修复报错！） ---------------------
depth = np.load(DEPTH_NPY)  # 正确：读取深度数组
# depth[depth <= 0] = np.nan  # 无效区域设为空，不显示

plt.figure(figsize=(10,5))
plt.imshow(depth, cmap='plasma', vmin=0, vmax=50)
plt.axis('off')
plt.savefig("depth.png", bbox_inches='tight', dpi=200)
plt.close()

# --------------------- 5. 3D 点云（固定视角，颜色正常） ---------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    xyz_vehicle[:,0], xyz_vehicle[:,1], xyz_vehicle[:,2],
    c=rgb, s=0.5, alpha=0.8
)

ax.view_init(elev=15, azim=-90)
ax.set_box_aspect([40, 50, 8])
ax.set_xlim(-20, 20)
ax.set_ylim(0, 50)
ax.set_zlim(-3, 5)
plt.axis('off')
plt.savefig("lidar_3d.png", dpi=300)
plt.close()

print("✅ 全部生成完成！无报错！")