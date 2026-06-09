#!/bin/bash

# MagicDrive-V2 多卡评测脚本
# 使用方法: bash scripts/eval_magicdrive_dist.sh [GPU数量]

# 默认使用 4 卡
NUM_GPUS=8

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 模型路径
CKPT_PATH=/mnt/ljy/MagicDrive-V2/ckpts/MagicDriveDiT-stage3-40k-ft

# 输出目录
OUTPUT_DIR=output/eval

# 配置文件
CONFIG=configs/magicdrive/test/17-16x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_fsp8_cfg2.0.py

echo "=========================================="
echo "MagicDrive-V2 多卡评测"
echo "GPU 数量: $NUM_GPUS"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 使用 torchrun 启动多卡评测
torchrun --standalone --nproc_per_node $NUM_GPUS \
    scripts/eval_magicdrive.py \
    $CONFIG \
    --ckpt-path $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples 10 \
    --view_idx 1
