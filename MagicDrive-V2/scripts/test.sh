export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # for GPU
export CUDA_VISIBLE_DEVICES=1,2,3  # for multi-GPU training
torchrun --standalone --nproc_per_node 3 scripts/test_magicdrive.py \
    configs/magicdrive/test/17-16x848x1600_stdit3_CogVAE_boxTDS_wCT_xCE_wSST_map0_fsp8_cfg2.0.py \
    --cfg-options model.from_pretrained=/mnt/ljy/MagicDrive-V2/ckpts/MagicDriveDiT-stage3-40k-ft tag=nuscenes_test