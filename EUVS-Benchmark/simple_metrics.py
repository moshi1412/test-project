#!/usr/bin/env python3
"""简化版指标评估脚本 - 只需渲染图和真值图"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms.functional as tf
from PIL import Image
import lpips
import numpy as np

def calculate_psnr(img1, img2):
    """计算峰值信噪比"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()

def calculate_ssim(img1, img2):
    """计算结构相似性"""
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    img1 = img1.float()
    img2 = img2.float()
    
    kernel = torch.ones(11, 11) / 121
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(img1.device)
    
    mu1 = torch.nn.functional.conv2d(img1, kernel, padding=5)
    mu2 = torch.nn.functional.conv2d(img2, kernel, padding=5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, kernel, padding=5) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, kernel, padding=5) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, kernel, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def evaluate_simple(gt_path, renders_path, output_json_path):
    """简化版评估函数"""
    # 初始化设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化 LPIPS 模型
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # 获取图像文件列表
    render_files = sorted([f for f in os.listdir(renders_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gt_files = sorted([f for f in os.listdir(gt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(render_files)} render images and {len(gt_files)} ground truth images")
    
    # 确保文件数量匹配
    min_count = min(len(render_files), len(gt_files))
    render_files = render_files[:min_count]
    gt_files = gt_files[:min_count]
    
    ssims, psnrs, lpipss = [], [], []
    
    with torch.no_grad():
        for render_file, gt_file in tqdm(zip(render_files, gt_files), total=min_count, desc="Evaluating"):
            # 读取图像
            render = Image.open(os.path.join(renders_path, render_file)).convert('RGB')
            gt = Image.open(os.path.join(gt_path, gt_file)).convert('RGB')
            
            # 转换为张量
            render_tensor = tf.to_tensor(render).unsqueeze(0).to(device)
            gt_tensor = tf.to_tensor(gt).unsqueeze(0).to(device)
            
            # 调整尺寸以匹配
            min_height = min(render_tensor.shape[2], gt_tensor.shape[2])
            min_width = min(render_tensor.shape[3], gt_tensor.shape[3])
            render_tensor = torch.nn.functional.interpolate(render_tensor, size=(min_height, min_width))
            gt_tensor = torch.nn.functional.interpolate(gt_tensor, size=(min_height, min_width))
            
            # 计算指标
            psnr = calculate_psnr(render_tensor[0], gt_tensor[0])
            ssim = calculate_ssim(render_tensor, gt_tensor)
            lpips_score = lpips_model(render_tensor, gt_tensor).item()
            
            ssims.append(ssim)
            psnrs.append(psnr)
            lpipss.append(lpips_score)
    
    # 计算平均值
    results = {
        "SSIM": sum(ssims) / len(ssims),
        "PSNR": sum(psnrs) / len(psnrs),
        "LPIPS": sum(lpipss) / len(lpipss),
        "num_images": len(ssims)
    }
    
    # 保存结果
    with open(output_json_path, 'w') as fp:
        json.dump(results, fp, indent=True)
    
    print("\n=== 评估结果 ===")
    print(f"SSIM: {results['SSIM']:.4f}")
    print(f"PSNR: {results['PSNR']:.2f}")
    print(f"LPIPS: {results['LPIPS']:.4f}")
    print(f"图像数量: {results['num_images']}")
    print(f"结果已保存到: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简化版图像质量评估脚本")
    parser.add_argument('--gt_path', '-g', type=str, required=True,
                        help="真值图像目录路径")
    parser.add_argument('--renders_path', '-r', type=str, required=True,
                        help="渲染图像目录路径")
    parser.add_argument('--output_json_path', '-o', type=str, required=True,
                        help="输出结果JSON文件路径")
    
    args = parser.parse_args()
    
    evaluate_simple(
        gt_path=args.gt_path,
        renders_path=args.renders_path,
        output_json_path=args.output_json_path
    )
