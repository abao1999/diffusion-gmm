CUDA_VISIBLE_DEVICES=4 python scripts/edm_generate.py \
    --save_dir=$WORK/vision_datasets/edm_imagenet64_more/volcano \
    --class=980 \
    --seeds=0-64 \
    --batch=64 \
    --steps=256 \
    --S_churn=40 \
    --S_min=0.05 \
    --S_max=50 \
    --S_noise=1.003 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
    --snapshot_save_dir=$WORK/vision_datasets/edm_imagenet64_snapshots/volcano \
    --snapshot_interval=32 \
    --n_snapshot_images_save_per_batch=1

# CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 generate.py \
#     --save_dir=$WORK/vision_datasets/edm_imagenet64_more/goldfinch \
#     --class=11 \
#     --seeds=0-1023 \
#     --batch=64 \
#     --steps=256 \
#     --S_churn=40 \
#     --S_min=0.05 \
#     --S_max=0.625 \
#     --S_noise=1.003 \
#     --sigma_min=0.002 \
#     --sigma_max=1 \
#     --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
#     --snapshot_save_dir=$WORK/vision_datasets/edm_imagenet64_snapshots/goldfinch \
#     --snapshot_interval=32 \
#     --n_snapshot_images_save_per_batch=1

# CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 generate.py \
#     --outdir="$WORK/vision_datasets/edm_imagenet64_all/volcano" \
#     --class=980 \
#     --seeds=0-10239 \
#     --batch=64 \
#     --steps=256 \
#     --S_churn=40 \
#     --S_min=0.05 \
#     --S_max=50 \
#     --S_noise=1.003 \
#     --network="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"