#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision.transforms.functional as tf
import json
from PIL import Image
from pathlib import Path

from tqdm import tqdm
from lib.config import cfg
from lib.utils.loss_utils import ssim, psnr
from lib.utils.lpipsPyTorch import lpips
from lib.datasets.dataset import Dataset


def evaluate(split='test'):
    print(split)
    scene_dir = cfg.model_path
    dataset = Dataset(scene_idx='003')
    if split == 'test':
        test_dir = Path(scene_dir) / "test"
        cam_infos = dataset.test_cameras[1]
    else:
        test_dir = Path(scene_dir) / "train"
        cam_infos = dataset.train_cameras[1]
        
    cam_infos = list(sorted(cam_infos, key=lambda x: x.id))
    
    full_dict = {}
    per_view_dict = {}
    
    print(f"Scene: {scene_dir }")
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    
    for method in os.listdir(test_dir):
        print("Method:", method)
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}

        renders = []
        gts = []
        image_names = []

        for cam_info in tqdm(cam_infos, desc="Reading image progress"):
            image_name = cam_info.image_name
            render_path = test_dir / method / f'{image_name}_rgb.png'
            gt_path = test_dir / method / f'{image_name}_gt.png'
            
            render = Image.open(render_path)
            gt = Image.open(gt_path)
            renders.append(tf.to_tensor(render)[:3, :, :])
            gts.append(tf.to_tensor(gt)[:3, :, :])
            image_names.append(image_name)

        psnrs = []
        ssims = []
        lpipss = []
        valid_names = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            render = renders[idx].cuda()
            gt = gts[idx].cuda()
            name = image_names[idx]
            
            ssim_val = ssim(render, gt)
            psnr_val = psnr(render, gt)
            lpips_val = lpips(render, gt, net_type='alex')

            # ====================== 核心修复 ======================
            # 如果 PSNR 是 inf，直接跳过，不加入平均
            if torch.isinf(psnr_val):
                print(f"跳过 {name} (PSNR=inf)")
                continue
            # ======================================================
            
            ssims.append(ssim_val)
            psnrs.append(psnr_val)
            lpipss.append(lpips_val)
            valid_names.append(name)
        
        print(f"有效样本数：{len(psnrs)} / 总样本数：{len(image_names)}")
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
        print("")
        
        full_dict[scene_dir][method].update({
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item()
        })
        per_view_dict[scene_dir][method].update({
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), valid_names)},
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), valid_names)},
            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), valid_names)}
        })

    with open(scene_dir + f"/results_{split}.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + f"/per_view_{split}.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    if cfg.eval.eval_train:
        evaluate(split='train')
    if cfg.eval.eval_test:
        evaluate(split='test')