#!/bin/bash

# 填入你自己的hf token
export HF_TOKEN="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"
# 开启hf-mirror加速
export HF_ENDPOINT=https://hf-mirror.com
# 高速下载开关（可选，提速）
export HF_HUB_ENABLE_HF_TRANSFER=0

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='deepvk/bert-base-uncased',
    local_dir='./bert-base-uncased',
    resume_download=True,
)
"