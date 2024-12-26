# CUDA_VISIBLE_DEVICES=4 python scripts/edm_generate.py \
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --standalone --nproc_per_node=3 scripts/edm_generate.py \
    edm.network_pkl=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
    edm.save_dir=$WORK/vision_datasets/edm_imagenet64_more/volcano \
    edm.class_idx=980 \
    edm.seeds=0-63 \
    edm.max_batch_size=64 \
    sampler.num_steps=256 \
    sampler.S_churn=40 \
    sampler.S_min=0.05 \
    sampler.S_max=50 \
    sampler.S_noise=1.003 \
    snapshots.save_dir=$WORK/vision_datasets/edm_imagenet64_snapshots/volcano \
    snapshots.interval=32 \
    snapshots.n_images_to_save_per_batch=1