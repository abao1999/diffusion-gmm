main_dir=$(dirname "$(dirname "$0")")

class_list=(
    "church"
    "tench"
    "english_springer"
    "french_horn"
)

class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
echo $class_list_json

model_id=resnet101

python scripts/save_cnn_representations.py \
    cnn.data_dir=$WORK/vision_datasets/real_imagenet_resize256 \
    cnn.model_id=${model_id} \
    cnn.class_list=${class_list_json} \
    cnn.save_dir=$WORK/vision_datasets/real_imagenet_representations/${model_id} \
    cnn.n_samples_per_class=1350 \
    cnn.batch_size=64 \
    cnn.num_workers=8 \
    cnn.device=cuda:2 \
    cnn.save_as_pt=false

python scripts/save_cnn_representations.py \
    cnn.data_dir=$WORK/vision_datasets/edm2_imagenet512_resize256 \
    cnn.model_id=${model_id} \
    cnn.class_list=${class_list_json} \
    cnn.save_dir=$WORK/vision_datasets/edm2_representations/${model_id} \
    cnn.n_samples_per_class=1350 \
    cnn.batch_size=64 \
    cnn.num_workers=8 \
    cnn.device=cuda:2 \
    cnn.save_as_pt=false