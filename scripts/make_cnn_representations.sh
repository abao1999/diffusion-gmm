main_dir=$(dirname "$(dirname "$0")")

class_list=(
    "baseball"
    "cauliflower"
    "church"
    "coral_reef"
    "english_springer"
    "french_horn"
    "garbage_truck"
    "goldfinch"
    "kimono"
    "mountain_bike"
    "patas_monkey"
    "pizza"
    "planetarium"
    "polaroid"
    "racer"
    "salamandra"
    "tabby"
    "tench"
    "trimaran"
    "volcano"
)

class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
echo $class_list_json

python scripts/save_cnn_representations.py \
    cnn.data_dir=$WORK/vision_datasets/edm_imagenet64_all \
    cnn.class_list=${class_list_json} \
    cnn.save_dir=$WORK/vision_datasets/representations \
    cnn.n_samples_per_class=10240 \
    cnn.batch_size=64 \
    cnn.num_workers=8 \
    cnn.device=cuda:2 \
    cnn.save_as_pt=false
