import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from utils.pcd_utils import fetchPly  # 用你现有工具读取PLY

# ====================== 你只需要改这里 ======================
PLY_PATH = "/mnt/ljy/street_crafter/data/waymo/000/lidar/background/000000.ply"
SAVE_BEV_PATH = "bev.png"
# ============================================================

# 1. 读取点云
ply = fetchPly(PLY_PATH)
xyz = ply.points[ply.mask]
rgb = ply.colors[ply.mask]

print(f"点云数量: {xyz.shape[0]}")
if xyz.shape[0] == 0:
    print("❌ 没有点！get_lidar 提取失败！")
    exit()

# 2. 提取 BEV (俯视 X-Y 平面)
xy = xyz[:, :2]
z = xyz[:, 2]

# 3. 范围裁剪（Waymo 标准范围）
mask = (
    (xy[:, 0] > -80) & (xy[:, 0] < 80) &
    (xy[:, 1] > -80) & (xy[:, 1] < 80)
)
xy = xy[mask]
z = z[mask]
rgb = rgb[mask] / 255.0

if len(xy) == 0:
    print("❌ 裁剪后无点，点云范围异常！")
    exit()

# 4. 绘制 BEV
plt.figure(figsize=(12, 12))
plt.scatter(xy[:, 0], xy[:, 1], c=rgb, s=0.5, alpha=0.8)
plt.axis('equal')
plt.title("Waymo LiDAR BEV 鸟瞰图")
plt.tight_layout()
plt.savefig(SAVE_BEV_PATH, dpi=300)
plt.close()

print(f"✅ BEV 图已保存: {SAVE_BEV_PATH}")
print("👉 如果图里能看到车道、车、规整点云 → 点云是对的！")