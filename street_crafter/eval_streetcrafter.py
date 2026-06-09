#!/usr/bin/env python3
"""
StreetCrafter 评测脚本
计算 PSNR、SSIM、LPIPS 指标
"""
import os
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

def load_image(path):
    """加载图像并转换为 RGB 张量"""
    image = Image.open(path).convert('RGB').resize((512, 512))
    img_np = np.array(image)
    return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

def compute_metrics(gt_path, render_path, lpips_model=None, device=None):
    """计算单张图像的指标"""
    gt_img = load_image(gt_path).to(device)
    render_img = load_image(render_path).to(device)
    
    # PSNR
    psnr_val = psnr(gt_img.permute(1, 2, 0).cpu().numpy(), render_img.permute(1, 2, 0).cpu().numpy(), data_range=1.0)
    
    # SSIM
    ssim_val = ssim(
        gt_img.permute(1, 2, 0).cpu().numpy(), 
        render_img.permute(1, 2, 0).cpu().numpy(), 
        data_range=1.0,
        channel_axis=2,
        win_size=11
    )
    
    # LPIPS
    lpips_val = None
    if lpips_model is not None:
        with torch.no_grad():
            lpips_val = lpips_model(gt_img.unsqueeze(0), render_img.unsqueeze(0)).item()
    
    return {
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'lpips': float(lpips_val) if lpips_val is not None else None
    }

def evaluate_streetcrafter(gt_dir, render_dir, output_json_path, max_samples=None):
    """评测 StreetCrafter 渲染质量"""
    
    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")
    
    # 初始化 LPIPS 模型
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    # 获取所有图像对
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('_0.png') and not f.endswith('_mask.png')])
    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('2.00.png') and not f.endswith('_mask.png')])
    
    print(f"GT 图像数量：{len(gt_files)}")
    print(f"渲染图像数量：{len(render_files)}")
    
    # 限制样本数量
    if max_samples is not None:
        gt_files = gt_files[:max_samples]
    
    # 计算指标
    all_psnr = []
    all_ssim = []
    all_lpips = []
    results = []
    
    for gt_file in gt_files:
        render_file = gt_file.replace('.png', '_shift_2.00.png')
        # if render_file not in render_files:
        #     print(f"警告：{render_file} 在渲染目录中不存在")
        #     continue
        
        gt_path = os.path.join(gt_dir, gt_file)
        render_path = os.path.join(render_dir, render_file)
        
        try:
            metrics = compute_metrics(gt_path, render_path, lpips_model, device)
            
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            if metrics['lpips'] is not None:
                all_lpips.append(metrics['lpips'])
            
            results.append({
                'gt_file': gt_file,
                'render_file': render_file,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'lpips': metrics['lpips']
            })
            
            if len(results) % 10 == 0:
                print(f"已处理 {len(results)} 张图像")
                
        except Exception as e:
            print(f"处理 {gt_file} 时出错：{e}")
            continue
    
    # 计算平均指标（全部转成 Python 原生 float）
    avg_psnr = float(sum(all_psnr) / len(all_psnr)) if all_psnr else -1.0
    avg_ssim = float(sum(all_ssim) / len(all_ssim)) if all_ssim else -1.0
    avg_lpips = float(sum(all_lpips) / len(all_lpips)) if all_lpips else -1.0
    
    # 汇总结果（全部转成 float）
    summary = {
        'num_samples': len(results),
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_lpips': avg_lpips,
        'std_psnr': float(np.std(all_psnr)) if all_psnr else -1.0,
        'std_ssim': float(np.std(all_ssim)) if all_ssim else -1.0,
        'std_lpips': float(np.std(all_lpips)) if all_lpips else -1.0,
    }
    
    # 保存结果
    output_data = {
        'summary': summary,
        'detailed_results': results,
        'config': {
            'gt_dir': gt_dir,
            'render_dir': render_dir,
            'max_samples': max_samples
        }
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # 打印结果
    print("\n" + "="*60)
    print("StreetCrafter 评测结果")
    print("="*60)
    print(f"样本数量：{len(results)}")
    print(f"平均 PSNR: {avg_psnr:.4f} ± {summary['std_psnr']:.4f}")
    print(f"平均 SSIM: {avg_ssim:.4f} ± {summary['std_ssim']:.4f}")
    if avg_lpips >= 0:
        print(f"平均 LPIPS: {avg_lpips:.4f} ± {summary['std_lpips']:.4f}")
    print(f"结果已保存到：{output_json_path}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="StreetCrafter 评测脚本")
    parser.add_argument('--gt_dir', type=str, default='/mnt/ljy/street_crafter/data/waymo/training_set_processed/049/images', 
                       help='Ground Truth 图像目录')
    parser.add_argument('--render_dir', type=str,default='/mnt/ljy/street_crafter/output/waymo/waymo_val_049/diffusion', 
                       help='渲染图像目录')
    parser.add_argument('--output_json_path', type=str, 
                       default='/mnt/ljy/street_crafter/output/waymo/waymo_val_049/streetcrafter_metrics.json',
                       help='输出 JSON 文件路径')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大评测样本数量（None 表示全部）')
    
    args = parser.parse_args()
    
    evaluate_streetcrafter(
        gt_dir=args.gt_dir,
        render_dir=args.render_dir,
        output_json_path=args.output_json_path,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()