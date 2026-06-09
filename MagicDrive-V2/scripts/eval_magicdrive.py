#!/usr/bin/env python3
"""
MagicDrive-V2 评测脚本
生成可用于 EUVS-Benchmark/simple_metrics.py 的 GT 和渲染图像
参考 test_magicdrive.py 实现
"""
import os
import sys
import copy
from pprint import pformat
from functools import partial

sys.path.append(".")
DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

import torch
if not torch.cuda.is_available() or DEVICE_TYPE == 'npu':
    USE_NPU = True
    os.environ['DEVICE_TYPE'] = "npu"
    DEVICE_TYPE = "npu"
    print("Enable NPU!")
    try:
        import xformers
        import xformers.ops
    except Exception as e:
        print(f"Got {e} during import xformers!")
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
else:
    USE_NPU = False
import magicdrivedit.utils.module_contrib

import colossalai
import torch.distributed as dist
import torchvision.transforms as TF
from einops import rearrange, repeat
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from mmengine.runner import set_random_seed
from tqdm import tqdm
from mmcv.parallel import DataContainer
from PIL import Image

from magicdrivedit.acceleration.communications import gather_tensors, serialize_state, deserialize_state
from magicdrivedit.acceleration.parallel_states import (
    set_sequence_parallel_group,
    get_sequence_parallel_group,
    set_data_parallel_group,
    get_data_parallel_group,
)
from magicdrivedit.datasets import save_sample
from magicdrivedit.datasets.dataloader import prepare_dataloader
from magicdrivedit.registry import DATASETS, MODELS, SCHEDULERS, build_module
from magicdrivedit.utils.config_utils import parse_configs, define_experiment_workspace, save_training_config, merge_dataset_cfg, mmengine_conf_get, mmengine_conf_set
from magicdrivedit.utils.inference_utils import (
    concat_6_views_pt,
    add_null_condition,
    enable_offload,
)
from magicdrivedit.utils.misc import (
    reset_logger,
    is_distributed,
    to_torch_dtype,
    collate_bboxes_to_maxlen,
    move_to,
    add_box_latent,
)
from magicdrivedit.utils.train_utils import sp_vae

VIEW_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def make_file_dirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_image_tensor(image_tensor, save_path):
    """保存图像张量为 PNG 文件"""
    image_tensor = (image_tensor * 255).clamp(0, 255).byte().cpu()
    image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path, format='PNG')


def main():
    torch.set_grad_enabled(False)
    
    # ======================================================
    # 先解析自定义评测参数
    # ======================================================
    import argparse
    parser = argparse.ArgumentParser(description="MagicDrive-V2 评测脚本")
    parser.add_argument('config', type=str, help="配置文件路径")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录")
    parser.add_argument('--num_samples', type=int, default=10, help="评估样本数量")
    parser.add_argument('--view_idx', type=int, default=1, help="选择相机视角 (0-5, 默认 CAM_FRONT)")
    # 添加 parse_configs 需要的参数
    parser.add_argument('--ckpt-path', default='/mnt/ljy/MagicDrive-V2/ckpts/MagicDriveDiT-stage3-40k-ft', type=str, help="path to model ckpt")
    parser.add_argument('--cfg-options', nargs='+', help='override some settings in the used config')
    
    args = parser.parse_args()
    
    # 将参数重新注入 sys.argv 供 parse_configs 使用
    sys.argv = [sys.argv[0], args.config]
    if args.ckpt_path is not None:
        sys.argv.extend(['--ckpt-path', args.ckpt_path])
    if args.cfg_options is not None:
        sys.argv.extend(['--cfg-options'] + args.cfg_options)
    
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == dataset config ==
    if cfg.num_frames is None:
        num_data_cfgs = len(cfg.data_cfg_names)
        datasets = []
        val_datasets = []
        for (res, data_cfg_name), overrides in zip(
                cfg.data_cfg_names, cfg.get("dataset_cfg_overrides", [[]] * num_data_cfgs)):
            dataset, val_dataset = merge_dataset_cfg(cfg, data_cfg_name, overrides)
            datasets.append((res, dataset))
            val_datasets.append((res, val_dataset))
        dataset = {"type": "NuScenesMultiResDataset", "cfg": datasets}
        val_dataset = {"type": "NuScenesMultiResDataset", "cfg": val_datasets}
    else:
        dataset, val_dataset = merge_dataset_cfg(
            cfg, cfg.data_cfg_name, cfg.get("dataset_cfg_overrides", []),
            cfg.num_frames)
    cfg.dataset = val_dataset
    
    # 设置 collate 参数
    if hasattr(cfg.dataset, "img_collate_param"):
        cfg.dataset.img_collate_param.is_train = False
    else:
        for d in cfg.dataset.cfg:
            d[1].img_collate_param.is_train = False
    
    cfg.batch_size = 1
    cfg.ignore_ori_imgs = False
    cfg.use_back_trans = True
    cfg.save_mode = "single-view"
    cfg.use_map0 = cfg.get("use_map0", False)

    # == 设备和数据类型 ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "bf16")
    dtype = to_torch_dtype(cfg_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if USE_NPU:
        if mmengine_conf_get(cfg, "text_encoder.shardformer", None):
            mmengine_conf_set(cfg, "text_encoder.shardformer", False)
        if mmengine_conf_get(cfg, "model.bbox_embedder_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.bbox_embedder_param.enable_xformers", False)
        if mmengine_conf_get(cfg, "model.frame_emb_param.enable_xformers", None):
            mmengine_conf_set(cfg, "model.frame_emb_param.enable_xformers", False)

    # == 初始化分布式环境 ==
    cfg.sp_size = cfg.get("sp_size", 1)
    if is_distributed():
        colossalai.launch_from_torch({})
    else:
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0,
            init_method="tcp://localhost:12355")
    coordinator = DistCoordinator()
    if cfg.sp_size > 1:
        DP_AXIS, SP_AXIS = 0, 1
        dp_size = dist.get_world_size() // cfg.sp_size
        pg_mesh = ProcessGroupMesh(dp_size, cfg.sp_size)
        dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
        sp_group = pg_mesh.get_group_along_axis(SP_AXIS)
        set_sequence_parallel_group(sp_group)
    else:
        dp_group = dist.group.WORLD
    set_data_parallel_group(dp_group)
    set_random_seed(seed=cfg.get("seed", 1024))

    # == 创建输出目录 ==
    gt_dir = os.path.join(args.output_dir, "gt")
    render_dir = os.path.join(args.output_dir, "render")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)
    print(f"输出目录: {args.output_dir}")
    print(f"  - GT 图像: {gt_dir}")
    print(f"  - 渲染图像: {render_dir}")

    # ======================================================
    # 构建数据集和数据加载器
    # ======================================================
    print("Building dataset...")
    dataset = build_module(cfg.dataset, DATASETS)
    dataset = torch.utils.data.Subset(dataset, list(range(min(args.num_samples, len(dataset)))))
    print(f"数据集包含 {len(dataset)} 个样本")

    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 1),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )

    def collate_data_container_fn(batch, *, collate_fn_map=None):
        return batch
    torch.utils.data._utils.collate.default_collate_fn_map.update({
        DataContainer: collate_data_container_fn
    })

    # ======================================================
    # 构建模型 & 加载权重
    # ======================================================
    print("Building models...")
    
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == 后处理变换 ==
    if cfg.use_back_trans:
        back_trans = TF.Compose([
            TF.Resize(cfg.post.resize, interpolation=TF.InterpolationMode.BICUBIC),
            TF.Pad(cfg.post.padding),
        ])
        cut_length = cfg.post.get("cut_length", None)
    else:
        def back_trans(x): return x
        cut_length = cfg.post.get("cut_length", None)

    # == 构建扩散模型 ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=(None, None, None),
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=cfg.sp_size > 1,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder

    # == 构建 scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # 推理
    # ======================================================
    cfg.cpu_offload = cfg.get("cpu_offload", False)
    if cfg.cpu_offload:
        text_encoder.t5.model.to("cpu")
        model.to("cpu")
        vae.to("cpu")
        text_encoder.t5.model, model, vae, last_hook = enable_offload(
            text_encoder.t5.model, model, vae, device)

    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)

    generator = torch.Generator("cpu").manual_seed(cfg.seed)
    bl_generator = torch.Generator("cpu").manual_seed(cfg.seed)

    image_idx = 0
    
    with tqdm(enumerate(dataloader), desc="Generating", total=len(dataloader)) as pbar:
        for i, batch in pbar:
            this_token = batch['meta_data']['metas'][0][0].data['token']
            
            if cfg.ignore_ori_imgs:
                B, T, NC = 1, *batch["pixel_values_shape"][0].tolist()[:2]
                latent_size = vae.get_latent_size(
                    (T, *batch["pixel_values_shape"][0].tolist()[-2:]))
            else:
                B, T, NC = batch["pixel_values"].shape[:3]
                latent_size = vae.get_latent_size((T, *batch["pixel_values"].shape[-2:]))

            # == 准备输入数据 ==
            print("prepare input data")
            y = batch.pop("captions")[0]
            maps = batch.pop("bev_map_with_aux").to(device, dtype)
            bbox = batch.pop("bboxes_3d_data")
            bbox = [bbox_i.data for bbox_i in bbox]
            bbox = collate_bboxes_to_maxlen(bbox, device, dtype, NC, T)
            cams = batch.pop("camera_param").to(device, dtype)
            cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")
            rel_pos = batch.pop("frame_emb").to(device, dtype)
            rel_pos = repeat(rel_pos, "B T ... -> (B NC) T 1 ...", NC=NC)

            # == 模型输入参数 ==
            model_args = {}
            model_args["maps"] = maps
            model_args["bbox"] = bbox
            model_args["cams"] = cams
            model_args["rel_pos"] = rel_pos
            model_args["fps"] = batch.pop('fps')
            model_args["height"] = batch.pop("height")
            model_args["width"] = batch.pop("width")
            model_args["num_frames"] = batch.pop("num_frames")
            model_args = move_to(model_args, device=device, dtype=dtype)
            model_args["mv_order_map"] = cfg.get("mv_order_map")
            model_args["t_order_map"] = cfg.get("t_order_map")

            for ns in range(num_sample):
                z = torch.randn(
                    len(y), vae.out_channels * NC, *latent_size, generator=generator,
                ).to(device=device, dtype=dtype)

                # == 采样 box latent ==
                if bbox is not None:
                    bbox = add_box_latent(bbox, B, NC, T,
                                          partial(model.sample_box_latent, generator=bl_generator))
                    new_bbox = {}
                    for k, v in bbox.items():
                        new_bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")
                    model_args["bbox"] = move_to(new_bbox, device=device, dtype=dtype)

                # == 添加 null condition ==
                if cfg.scheduler.type == "dpm-solver" and cfg.scheduler.cfg_scale == 1.0 or (
                    cfg.scheduler.type in ["rflow-slice",]
                ):
                    _model_args = copy.deepcopy(model_args)
                else:
                    _model_args = add_null_condition(
                        copy.deepcopy(model_args),
                        model.camera_embedder.uncond_cam.to(device),
                        model.frame_embedder.uncond_cam.to(device),
                        prepend=(cfg.scheduler.type == "dpm-solver"),
                        use_map0=cfg.get("use_map0", False),
                    )

                # == 推理 ==
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=y,
                    device=device,
                    additional_args=_model_args,
                    progress=False,
                    mask=None,
                )
                samples = rearrange(samples, "B (C NC) T ... -> (B NC) C T ...", NC=NC)
                if cfg.sp_size > 1:
                    samples = sp_vae(
                        samples.to(dtype),
                        partial(vae.decode, num_frames=_model_args["num_frames"]),
                        get_sequence_parallel_group(),
                    )
                else:
                    samples = vae.decode(samples.to(dtype), num_frames=_model_args["num_frames"])
                samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
                if cfg.cpu_offload:
                    last_hook.offload()
                samples = samples[:, :, :, slice(None, cut_length)]

                # == 保存生成图像 ==
                if coordinator.is_master():
                    gen_video = samples[0, args.view_idx]  # C, T, H, W
                    gen_video = back_trans(gen_video)
                    
                    for t in range(gen_video.shape[1]):
                        frame = gen_video[:, t]  # C, H, W
                        save_path = os.path.join(render_dir, f"{image_idx:06d}.png")
                        save_image_tensor(frame, save_path)
                        image_idx += 1

            # == 保存 GT 图像 ==
            if not cfg.ignore_ori_imgs:
                x = batch.pop("pixel_values").to(device, dtype)
                x = rearrange(x, "B T NC C ... -> B NC C T ...")
                x = x[:, :, :, slice(None, cut_length)]
                
                if coordinator.is_master():
                    gt_video = x[0, args.view_idx]  # C, T, H, W
                    gt_video = back_trans(gt_video)
                    
                    # 确保 GT 和生成图像数量一致
                    for t in range(min(gt_video.shape[1], gen_video.shape[1])):
                        frame = gt_video[:, t]
                        save_path = os.path.join(gt_dir, f"{image_idx - gen_video.shape[1] + t:06d}.png")
                        save_image_tensor(frame, save_path)

            coordinator.block_all()
    
    print(f"\n评测完成!")
    print(f"GT 图像: {gt_dir}")
    print(f"渲染图像: {render_dir}")
    print(f"生成图像数量: {image_idx}")
    print(f"\n使用 EUVS-Benchmark 评估:")
    print(f"python /mnt/ljy/EUVS-Benchmark/simple_metrics.py \\\n    --gt_path {gt_dir} \\\n    --renders_path {render_dir} \\\n    --output_json_path {os.path.join(args.output_dir, 'results.json')}")
    
    coordinator.destroy()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    main()
