#!/bin/bash
set -e

# ========== 配置项：替换为你的HF Read Token ==========
HF_TOKEN="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"
LOCAL_SAVE_DIR="./data/"
# ======================================================

# 开启HF国内镜像加速
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN}"
# 开启LFS高速传输
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== 开始下载 flymin/MagicDriveDiT-nuScenes-metadata ==="
echo "本地保存路径: ${LOCAL_SAVE_DIR}"

# Python huggingface_hub 下载（多线程、断点续传）
python -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='flymin/MagicDriveDiT-nuScenes-metadata',
    repo_type='dataset',
    local_dir='${LOCAL_SAVE_DIR}',
    token='${HF_TOKEN}',
    resume_download=True,
    max_workers=10,
    local_dir_use_symlinks=False
)
print('✅ MagicDriveDiT nuScenes metadata 下载全部完成!')
"