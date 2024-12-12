CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 generate.py \
    --save_dir=$WORK/vision_datasets/edm_imagenet64_more/goldfinch \
    --class=11 \
    --seeds=2048-10239 \
    --batch=64 \
    --steps=256 \
    --S_churn=40 \
    --S_min=0.05 \
    --S_max=50 \
    --S_noise=1.003 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
    --snapshot_save_dir=$WORK/vision_datasets/edm_imagenet64_snapshots/goldfinch \
    --snapshot_interval=32 \
    --n_snapshot_images_save_per_batch=1

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