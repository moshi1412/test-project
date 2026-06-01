#!/bin/bash

# ====================== 填写你的 HuggingFace token ======================
HF_TOKEN="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"
# ======================================================================

# 开启镜像加速
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="$HF_TOKEN"

# 下载 EUVS Benchmark（带权限、高速、断点续传）
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ai4ce/EUVS-Benchmark',
    repo_type='dataset',
    local_dir='./data/EUVS-Benchmark',
    resume_download=True,
    max_workers=8,
    token='$HF_TOKEN'
)
print('✅ EUVS-Benchmark 下载完成！')
"