#!/usr/bin/env python3
"""
复制 street_gaussians/data 中的 sky_mask 到 EmerNeRF/data 对应目录
"""
import os
import shutil
from pathlib import Path

# 源目录和目标目录
SOURCE_DIR = "/mnt/ljy/street_gaussians/data"
TARGET_DIR = "/mnt/ljy/EmerNeRF/data/waymo/processed/static32/training"

def copy_sky_masks():
    source_path = Path(SOURCE_DIR)
    target_path = Path(TARGET_DIR)

    # 获取所有 scene_id 文件夹
    scene_ids = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    print(f"找到 {len(scene_ids)} 个场景")

    copied_count = 0
    skipped_count = 0

    for scene_id in scene_ids:
        source_sky_mask = source_path / scene_id / "sky_mask"
        target_sky_mask = target_path / scene_id / "sky_mask"

        # 检查源目录是否存在
        if not source_sky_mask.exists():
            print(f"警告: {source_sky_mask} 不存在，跳过")
            skipped_count += 1
            continue

        # 检查目标父目录是否存在
        if not target_sky_mask.parent.exists():
            print(f"警告: {target_sky_mask.parent} 不存在，跳过 scene {scene_id}")
            skipped_count += 1
            continue

        # 创建目标目录
        target_sky_mask.mkdir(parents=True, exist_ok=True)

        # 复制所有 png 文件
        sky_mask_files = list(source_sky_mask.glob("*.png"))
        for src_file in sky_mask_files:
            dst_file = target_sky_mask / src_file.name
            shutil.copy2(src_file, dst_file)

        print(f"场景 {scene_id}: 复制了 {len(sky_mask_files)} 个 sky_mask 文件")
        copied_count += 1

    print(f"\n完成! 成功复制 {copied_count} 个场景, 跳过 {skipped_count} 个场景")

if __name__ == "__main__":
    copy_sky_masks()
