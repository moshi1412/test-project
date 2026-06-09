#!/usr/bin/env python3
"""
MagicDrive-V2 视频评测脚本
计算 PSNR、SSIM、LPIPS 指标
支持多视角视频比较
"""
import os
import argparse
import json
import glob
from pathlib import Path

import torch
import numpy as np
import imageio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

VIEW_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

def load_video_frames(video_path):
    """加载视频帧"""
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        # 转换为 RGB 并归一化到 [0, 1]
        if frame.shape[-1] == 4:  # RGBA
            frame = frame[..., :3]
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    return frames

def compute_frame_metrics(gt_frame, gen_frame, lpips_model=None):
    """计算单帧指标"""
    # 确保尺寸相同
    min_h = min(gt_frame.shape[1], gen_frame.shape[1])
    min_w = min(gt_frame.shape[2], gen_frame.shape[2])
    
    gt_frame = gt_frame[:, :min_h, :min_w]
    gen_frame = gen_frame[:, :min_h, :min_w]
    
    # PSNR
    psnr_val = float(psnr(gt_frame.permute(1, 2, 0).numpy(), gen_frame.permute(1, 2, 0).numpy(), data_range=1.0))
    
    # SSIM
    ssim_val = float(ssim(
        gt_frame.permute(1, 2, 0).numpy(), 
        gen_frame.permute(1, 2, 0).numpy(), 
        data_range=1.0,
        channel_axis=2,
        win_size=11
    ))
    
    # LPIPS
    lpips_val = None
    if lpips_model is not None:
        with torch.no_grad():
            # LPIPS 要求输入范围 [-1, 1]
            gt_input = (gt_frame.unsqueeze(0).cuda() * 2) - 1
            gen_input = (gen_frame.unsqueeze(0).cuda() * 2) - 1
            lpips_val = float(lpips_model(gt_input, gen_input).item())
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'lpips': lpips_val
    }

def evaluate_video_pair(gt_video_path, gen_video_path, lpips_model=None):
    """评估一对视频"""
    try:
        gt_frames = load_video_frames(gt_video_path)
        gen_frames = load_video_frames(gen_video_path)
        
        # 取最小帧数
        num_frames = min(len(gt_frames), len(gen_frames))
        
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        for i in range(num_frames):
            metrics = compute_frame_metrics(gt_frames[i], gen_frames[i], lpips_model)
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            if metrics['lpips'] is not None:
                all_lpips.append(metrics['lpips'])
        
        return {
            'num_frames': int(num_frames),
            'avg_psnr': float(sum(all_psnr) / len(all_psnr)) if all_psnr else -1.0,
            'avg_ssim': float(sum(all_ssim) / len(all_ssim)) if all_ssim else -1.0,
            'avg_lpips': float(sum(all_lpips) / len(all_lpips)) if all_lpips else -1.0,
            'std_psnr': float(np.std(all_psnr)) if all_psnr else -1.0,
            'std_ssim': float(np.std(all_ssim)) if all_ssim else -1.0,
            'std_lpips': float(np.std(all_lpips)) if all_lpips else -1.0,
        }
    
    except Exception as e:
        print(f"处理视频失败: {e}")
        return None

def evaluate_magicdrive(gt_dir, gen_dir, output_json_path, views=None, max_scenes=None):
    """评测 MagicDrive-V2 生成质量"""
    
    # 初始化 LPIPS 模型
    print("初始化 LPIPS 模型...")
    lpips_model = lpips.LPIPS(net='vgg').cuda()
    
    # 获取所有场景目录
    gt_scenes = sorted([d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))])
    gen_scenes = sorted([d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))])
    
    print(f"GT 场景数量：{len(gt_scenes)}")
    print(f"生成场景数量：{len(gen_scenes)}")
    
    # 限制场景数量
    if max_scenes is not None:
        gt_scenes = gt_scenes[:max_scenes]
    
    # 确定要评测的视角
    if views is None:
        views = VIEW_ORDER
    
    # 存储所有结果
    results = {}
    summary = {view: {'psnr': [], 'ssim': [], 'lpips': []} for view in views}
    
    scene_count = 0
    for scene_token in gt_scenes:
        # if scene_token not in gen_scenes:
        #     print(f"警告：{scene_token} 在生成目录中不存在")
        #     continue
        
        scene_results = {}
        
        for view in views:
            gt_video_path = os.path.join(gt_dir, scene_token, f"{scene_token}_{view}.mp4")
            gen_video_path = os.path.join(gen_dir, scene_token+'_gen3', f"{scene_token}_{view}.mp4")
            
            if not os.path.exists(gt_video_path):
                print(f"警告：GT 视频不存在: {gt_video_path}")
                continue
            if not os.path.exists(gen_video_path):
                print(f"警告：生成视频不存在: {gen_video_path}")
                continue
            
            metrics = evaluate_video_pair(gt_video_path, gen_video_path, lpips_model)
            if metrics is not None:
                scene_results[view] = metrics
                summary[view]['psnr'].append(metrics['avg_psnr'])
                summary[view]['ssim'].append(metrics['avg_ssim'])
                if metrics['avg_lpips'] >= 0:
                    summary[view]['lpips'].append(metrics['avg_lpips'])
        
        if scene_results:
            results[scene_token] = scene_results
            scene_count += 1
            
            if scene_count % 5 == 0:
                print(f"已处理 {scene_count} 个场景")
    
    # 计算汇总指标
    final_summary = {}
    for view in views:
        psnrs = summary[view]['psnr']
        ssims = summary[view]['ssim']
        lpipss = summary[view]['lpips']
        
        final_summary[view] = {
            'num_scenes': int(len(psnrs)),
            'avg_psnr': float(sum(psnrs) / len(psnrs)) if psnrs else -1.0,
            'avg_ssim': float(sum(ssims) / len(ssims)) if ssims else -1.0,
            'avg_lpips': float(sum(lpipss) / len(lpipss)) if lpipss else -1.0,
            'std_psnr': float(np.std(psnrs)) if psnrs else -1.0,
            'std_ssim': float(np.std(ssims)) if ssims else -1.0,
            'std_lpips': float(np.std(lpipss)) if lpipss else -1.0,
        }
    
    # 计算所有视角的平均
    all_psnrs = []
    all_ssims = []
    all_lpipss = []
    for view in views:
        all_psnrs.extend(summary[view]['psnr'])
        all_ssims.extend(summary[view]['ssim'])
        all_lpipss.extend(summary[view]['lpips'])
    
    final_summary['overall'] = {
        'num_scenes': int(scene_count),
        'num_views': int(len(views)),
        'avg_psnr': float(sum(all_psnrs) / len(all_psnrs)) if all_psnrs else -1.0,
        'avg_ssim': float(sum(all_ssims) / len(all_ssims)) if all_ssims else -1.0,
        'avg_lpips': float(sum(all_lpipss) / len(all_lpipss)) if all_lpipss else -1.0,
        'std_psnr': float(np.std(all_psnrs)) if all_psnrs else -1.0,
        'std_ssim': float(np.std(all_ssims)) if all_ssims else -1.0,
        'std_lpips': float(np.std(all_lpipss)) if all_lpipss else -1.0,
    }
    
    # 保存结果
    output_data = {
        'summary': final_summary,
        'detailed_results': results,
        'config': {
            'gt_dir': gt_dir,
            'gen_dir': gen_dir,
            'views': views,
            'max_scenes': max_scenes
        }
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # 打印结果
    print("\n" + "="*70)
    print("MagicDrive-V2 评测结果")
    print("="*70)
    print(f"评测场景数：{scene_count}")
    print(f"评测视角：{views}")
    print("-"*70)
    
    for view in views:
        vs = final_summary[view]
        print(f"\n{view}:")
        print(f"  PSNR: {vs['avg_psnr']:.4f} ± {vs['std_psnr']:.4f}")
        print(f"  SSIM: {vs['avg_ssim']:.4f} ± {vs['std_ssim']:.4f}")
        if vs['avg_lpips'] >= 0:
            print(f"  LPIPS: {vs['avg_lpips']:.4f} ± {vs['std_lpips']:.4f}")
    
    overall = final_summary['overall']
    print("\n" + "-"*70)
    print("综合指标 (所有视角平均):")
    print(f"  PSNR: {overall['avg_psnr']:.4f} ± {overall['std_psnr']:.4f}")
    print(f"  SSIM: {overall['avg_ssim']:.4f} ± {overall['std_ssim']:.4f}")
    if overall['avg_lpips'] >= 0:
        print(f"  LPIPS: {overall['avg_lpips']:.4f} ± {overall['std_lpips']:.4f}")
    print("="*70)
    print(f"结果已保存到：{output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="MagicDrive-V2 视频评测脚本")
    parser.add_argument('--gt_dir', type=str, default='outputs/eval/CogVAE-848-17f/MagicDriveSTDiT3-XL-2_17-16x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_fsp8_cfg2.0_nuscenes_test_20260604-1322/generation/gt_video',
                       help='Ground Truth 视频目录')
    parser.add_argument('--gen_dir', type=str, default='outputs/eval/CogVAE-848-17f/MagicDriveSTDiT3-XL-2_17-16x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_fsp8_cfg2.0_nuscenes_test_20260604-1322/generation/gen_video',
                       help='生成视频目录')
    parser.add_argument('--output_json_path', type=str,
                       default='/mnt/ljy/MagicDrive-V2/magicdrive_metrics.json',
                       help='输出 JSON 文件路径')
    parser.add_argument('--views', type=str, nargs='+',
                       default=['CAM_FRONT'],
                       help='要评测的视角列表')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='最大评测场景数量')
    
    args = parser.parse_args()
    
    # 验证视角
    valid_views = [v for v in args.views if v in VIEW_ORDER]
    if not valid_views:
        print(f"无效的视角，使用默认视角 CAM_FRONT")
        valid_views = ['CAM_FRONT']
    else:
        print(f"评测视角: {valid_views}")
    
    evaluate_magicdrive(
        gt_dir=args.gt_dir,
        gen_dir=args.gen_dir,
        output_json_path=args.output_json_path,
        views=valid_views,
        max_scenes=args.max_scenes
    )

if __name__ == "__main__":
    main()
