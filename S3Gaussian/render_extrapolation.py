#!/usr/bin/env python3
"""
S3Gaussian 轨迹外推渲染脚本

通过修改相机位置（在x/y方向上偏移指定距离）来渲染外推视角的图像。
直接从训练的 checkpoint (.pth) 加载模型参数，无需读取点云文件。

Usage:
    python render_extrapolation.py \
        --model_path /path/to/trained/model \
        --source_path /path/to/data \
        --offsets -2.0 0 2.0 \
        --output_dir ./output/extrapolation
"""

import os
import sys
import argparse
import torch
import numpy as np
import imageio
from argparse import Namespace
from tqdm import tqdm

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams
from gaussian_renderer import render
from utils.general_utils import safe_state


def create_offset_camera(cam, offset_x, offset_y, offset_z=0.0):
    """
    通过在世界坐标系中偏移相机位置来创建新相机。
    
    Args:
        cam: 原始Camera对象
        offset_x: X方向偏移 (左/右，单位: 米)
        offset_y: Y方向偏移 (前/后，单位: 米)
        offset_z: Z方向偏移 (上/下，单位: 米)
    
    Returns:
        修改后的Camera对象
    """
    # 获取当前相机中心在世界坐标系中的位置
    # camera_center 已经是世界坐标系中的位置
    cam_center = cam.camera_center.cpu().numpy()  # [3]
    
    print(f"Original camera center: {cam_center}")
    
    # 在世界坐标系中应用偏移
    new_cam_center = cam_center + np.array([offset_x, offset_y, offset_z])
    
    print(f"New camera center after offset ({offset_x}, {offset_y}, {offset_z}): {new_cam_center}")
    
    # 获取当前的 w2c 矩阵 (world_view_transform)
    w2c = cam.world_view_transform.cpu().numpy()  # [4, 4]
    
    # 求逆得到 c2w 矩阵
    c2w = np.linalg.inv(w2c)
    
    # 修改 c2w 中的相机位置
    c2w_new = c2w.copy()
    c2w_new[:3, 3] = new_cam_center
    
    # 求逆得到新的 w2c 矩阵
    w2c_new = np.linalg.inv(c2w_new)
    
    # 从新的 w2c 中提取旋转和平移
    # w2c = [R | t]，其中 R 是世界到相机的旋转，t 是平移
    R_new = w2c_new[:3, :3].T  # 存储为转置形式，与原始一致
    T_new = w2c_new[:3, 3]
    
    # 创建新相机
    new_cam = type(cam)(
        colmap_id=cam.colmap_id,
        R=R_new,
        T=T_new,
        FoVx=cam.FoVx,
        FoVy=cam.FoVy,
        image=cam.original_image,  # 使用同一张图像占位
        gt_alpha_mask=None,
        image_name=f"{cam.image_name}_offset_x{offset_x}_y{offset_y}",
        uid=cam.uid,
        trans=cam.trans,
        scale=cam.scale,
        data_device="cuda",
        sky_mask=cam.sky_mask,
        depth_map=cam.depth_map,
        semantic_mask=cam.semantic_mask,
        instance_mask=cam.instance_mask,
        num_panoptic_objects=cam.num_panoptic_objects,
        sam_mask=cam.sam_mask,
        dynamic_mask=cam.dynamic_mask,
        feat_map=cam.feat_map,
        objects=cam.objects,
        intrinsic=cam.intrinsic,
        c2w=c2w_new,
        time=cam.time
    )
    
    return new_cam


def load_model_from_checkpoint(model_path, gaussians, args):
    """
    从 checkpoint 加载模型参数
    """
    # 查找最新的 checkpoint
    checkpoint_path = os.path.join(model_path, "chkpnt_fine_50000.pth")
    if not os.path.exists(checkpoint_path):
        # 尝试查找其他 checkpoint
        import glob
        checkpoints = glob.glob(os.path.join(model_path, "chkpnt*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found in {model_path}")
        checkpoint_path = sorted(checkpoints)[-1]
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 加载 checkpoint
    # checkpoint 格式: (model_params, iteration)
    # model_params 是由 capture() 返回的 tuple
    checkpoint = torch.load(checkpoint_path)
    
    # 解析 checkpoint
    if isinstance(checkpoint, tuple) and len(checkpoint) >= 2:
        model_params = checkpoint[0]  # capture() 返回的 tuple
        iteration = checkpoint[1]
        print(f"Loaded checkpoint at iteration: {iteration}")
    else:
        # 兼容其他格式
        model_params = checkpoint
    
    # 恢复模型参数
    gaussians.restore(model_params, args)
    print("Model restored successfully!")


def render_extrapolation_views(scene, gaussians, pipe, bg_color, args):
    """渲染外推视角"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取相机列表
    if args.camera_split == "train":
        cameras = scene.getTrainCameras()
    elif args.camera_split == "test":
        cameras = scene.getTestCameras()
    else:  # full
        cameras = scene.getFullCameras()
    
    print(f"Number of {args.camera_split} cameras: {len(cameras)}")
    
    # 如果相机数量为0，尝试其他选项
    if len(cameras) == 0:
        print(f"Warning: No {args.camera_split} cameras found!")
        if args.camera_split != "full":
            print("Trying full camera list instead...")
            cameras = scene.getFullCameras()
            print(f"Number of full cameras: {len(cameras)}")
    
    if len(cameras) == 0:
        raise RuntimeError("No cameras found in the dataset!")
    
    # 解析偏移参数
    offsets = []
    for offset_str in args.offsets:
        offset = float(offset_str)
        offsets.append(offset)
    
    print(f"Will render with offsets (meters): {offsets}")
    
    # 渲染结果统计
    all_results = {}
    
    # 对每个偏移量进行渲染
    for offset in offsets:
        offset_name = f"offset_{offset:+.1f}m"
        print(f"\n{'='*60}")
        print(f"Rendering with offset: {offset:+.1f} meters")
        print(f"{'='*60}")
        
        offset_dir = os.path.join(args.output_dir, offset_name)
        os.makedirs(offset_dir, exist_ok=True)
        
        rendered_images = []
        offset_psnrs = []
        
        for idx, cam in enumerate(tqdm(cameras, desc=f"Offset {offset:+.1f}m")):
            # 创建偏移相机
            offset_cam = create_offset_camera(cam, offset_x=offset, offset_y=0.0)
            
            # 渲染
            with torch.no_grad():
                render_pkg = render(
                    offset_cam, gaussians, pipe, bg_color,
                    stage=args.stage,
                    return_dx=False,
                    render_feat=False
                )
            
            rendered_image = render_pkg["render"]
            
            # 保存渲染图
            if idx < args.max_save or args.max_save == -1:
                image_np = (rendered_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                image_path = os.path.join(offset_dir, f"{idx:04d}_{offset_cam.image_name}.png")
                imageio.imwrite(image_path, image_np)
            
            rendered_images.append(rendered_image)
            
            # 如果有GT图像，计算PSNR
            if cam.original_image is not None and args.compute_metrics:
                gt_image = cam.original_image
                mse = torch.mean((rendered_image - gt_image) ** 2)
                psnr = -10.0 * torch.log10(mse)
                offset_psnrs.append(psnr.item())
        
        # 统计结果
        result = {
            "offset": offset,
            "num_images": len(rendered_images),
        }
        if args.compute_metrics and len(offset_psnrs) > 0:
            result["mean_psnr"] = float(np.mean(offset_psnrs))
            result["std_psnr"] = float(np.std(offset_psnrs))
            print(f"Mean PSNR: {result['mean_psnr']:.4f} ± {result['std_psnr']:.4f}")
        
        all_results[offset_name] = result
        
        # 保存视频
        if args.save_video and len(rendered_images) > 0:
            print(f"Saving video for offset {offset:+.1f}m...")
            video_path = os.path.join(offset_dir, f"video_{offset:+.1f}m.mp4")
            with imageio.get_writer(video_path, fps=args.fps) as writer:
                for img in rendered_images:
                    img_np = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    writer.append_data(img_np)
            print(f"Video saved to: {video_path}")
    
    # 保存汇总结果
    import json
    summary_path = os.path.join(args.output_dir, "extrapolation_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="S3Gaussian 轨迹外推渲染")
    
    # Use the same parameter groups as training script
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    op = OptimizationParams(parser)  # 需要用于 training_setup
    
    # Extrapolation-specific parameters
    parser.add_argument("--offsets", nargs="+", default=["-2.0", "0.0", "2.0"],
                        help="Camera position offsets in meters (e.g., -2.0 0.0 2.0)")
    parser.add_argument("--output_dir", type=str, default="./output/extrapolation",
                        help="Output directory for extrapolated images")
    parser.add_argument("--max_save", type=int, default=20,
                        help="Maximum number of images to save per offset (use -1 for all)")
    parser.add_argument("--save_video", action="store_true", default=True,
                        help="Save video for each offset")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for output video")
    parser.add_argument("--compute_metrics", action="store_true", default=True,
                        help="Compute PSNR against original images")
    parser.add_argument("--stage", type=str, default="fine",
                        help="Training stage (fine or coarse)")
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--camera_split", type=str, default="full",
                        choices=["train", "test", "full"],
                        help="Which camera split to use for rendering")
    parser.add_argument("--use_fags",type=bool, default=False, help="use fags deformation")
    args = parser.parse_args(sys.argv[1:])
    
    # Extract parameters using the same method as training script
    model_args = lp.extract(args)
    pipeline_args = pp.extract(args)
    hidden_args = hp.extract(args)
    opt_args = op.extract(args)
    
    # Merge all arguments
    args = Namespace(**{**vars(model_args), **vars(pipeline_args), **vars(hidden_args), **vars(opt_args), **vars(args)})
    
    # Set random state
    safe_state(args.quiet)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Loading scene from: {args.source_path}")
    
    # Initialize Gaussian model
    gaussians = GaussianModel(args.sh_degree, args)
    
    # Load scene WITHOUT loading Gaussian parameters (load_iteration=None)
    # 这里只加载相机信息，不加载点云
    scene = Scene(
        args,
        gaussians,
        load_iteration=None,  # 关键：不加载点云
        load_coarse=False,
        bg_gaussians=None,
        build_octree=False,
        build_grid=False,
        build_featgrid=False
    )
    
    # 从 checkpoint 加载模型参数
    load_model_from_checkpoint(args.model_path, gaussians, args)
    
    # Background color
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Render extrapolation views
    results = render_extrapolation_views(scene, gaussians, pipeline_args, background, args)
    
    print("\n=== Summary ===")
    for name, result in results.items():
        print(f"{name}: {result}")


if __name__ == "__main__":
    main()
