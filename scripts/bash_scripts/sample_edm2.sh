# CUDA_VISIBLE_DEVICES=0 python scripts/edm2_generate.py \
#     edm2.preset=edm2-img512-l-guid-dino \
#     edm2.class_idx=217 \
#     edm2.seeds=0-15 \
#     edm2.max_batch_size=32 \
#     edm2.snapshot_save_dir=$WORK/vision_datasets/edm2_imagenet512_snapshots/english_springer \
#     edm2.outdir=$WORK/vision_datasets/edm2_imagenet512/english_springer

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/edm2_generate.py \
    edm2.preset=edm2-img512-l-guid-dino \
    edm2.class_idx=566 \
    edm2.seeds=1024-4095 \
    edm2.max_batch_size=32 \
    edm2.snapshot_save_dir=$WORK/vision_datasets/edm2_imagenet512_snapshots/french_horn \
    edm2.outdir=$WORK/vision_datasets/edm2_imagenet512/french_horn

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/edm2_generate.py \
    edm2.preset=edm2-img512-l-guid-dino \
    edm2.class_idx=217 \
    edm2.seeds=1024-4095 \
    edm2.max_batch_size=32 \
    edm2.snapshot_save_dir=$WORK/vision_datasets/edm2_imagenet512_snapshots/english_springer \
    edm2.outdir=$WORK/vision_datasets/edm2_imagenet512/english_springer

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/edm2_generate.py \
    edm2.preset=edm2-img512-l-guid-dino \
    edm2.class_idx=497 \
    edm2.seeds=1024-4095 \
    edm2.max_batch_size=32 \
    edm2.snapshot_save_dir=$WORK/vision_datasets/edm2_imagenet512_snapshots/church \
    edm2.outdir=$WORK/vision_datasets/edm2_imagenet512/church

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 scripts/edm2_generate.py \
    edm2.preset=edm2-img512-l-guid-dino \
    edm2.class_idx=0 \
    edm2.seeds=1024-4095 \
    edm2.max_batch_size=32 \
    edm2.snapshot_save_dir=$WORK/vision_datasets/edm2_imagenet512_snapshots/tench \
    edm2.outdir=$WORK/vision_datasets/edm2_imagenet512/tench