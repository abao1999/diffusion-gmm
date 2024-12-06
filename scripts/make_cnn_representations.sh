main_dir=$(dirname "$(dirname "$0")")

n_samples_per_class=2048
batch_size=64
num_workers=24

datetime=$(date +%m-%d_%H-%M-%S)

class_list=(
    "english_springer"
)
class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
echo $class_list_json

python scripts/save_cnn_representations.py \
    cnn.data_dir=$WORK/vision_datasets/edm_imagenet64_all \
    cnn.class_list=${class_list_json} \
    cnn.save_dir=$WORK/vision_datasets/representations \
    cnn.n_samples_per_class=1024 \
    cnn.batch_size=64 \
    cnn.num_workers=4 \
    cnn.device=cuda:2 \
    cnn.save_as_pt=false
