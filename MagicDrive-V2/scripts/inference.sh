# ${GPUS} can be 1/2/4/8 for sequence parallel.
# ${CFG} can be any file located in `configs/magicdrive/inference/`.
# ${PATH_TO_MODEL} can be path to `ema.pt` or path to `model` from the checkpoint.
# ${FRAME} can be 1/9/17/33/65/129/full...(8n+1). 1 for image; full for the full-length of nuScenes.
# `cpu_offload=true` and `scheduler.type=rflow-slice` can be omitted if you have enough GPU memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=1
CFG=/mnt/ljy/MagicDrive-V2/configs/magicdrive/inference/fullx848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py
PATH_TO_MODEL=/mnt/ljy/MagicDrive-V2/ckpts/MagicDriveDiT-stage3-40k-ft
FRAME=1

torchrun --standalone --nproc_per_node ${GPUS} scripts/inference_magicdrive.py ${CFG} \
    --cfg-options model.from_pretrained=${PATH_TO_MODEL} num_frames=${FRAME} \
    cpu_offload=true scheduler.type=rflow-slice