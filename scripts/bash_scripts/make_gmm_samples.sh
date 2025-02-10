main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets
n_samples_fit=4096
n_samples_generate=4096
sample_idx=0

dataset_name="edm2_representations"
# dataset_name="edm_imagenet64_all"
gmm_dataset_name="gmm_$dataset_name"
model_id=resnet101

save_dir=$data_dir/$gmm_dataset_name/${model_id}
stats_save_dir=$data_dir/computed_stats/${dataset_name}/${model_id}
# stats_save_dir=null
echo $save_dir
echo $stats_save_dir

rseed=99

class_list=(
    "church"
    "tench"
    "english_springer"
    "french_horn"
)

for i in "${!class_list[@]}"; do
    class_name=${class_list[i]}
    echo $class_name
    python scripts/sample_gmm.py \
        gmm.n_components=1 \
        gmm.data_dir=$data_dir/$dataset_name/${model_id} \
        gmm.covariance_type=full \
        gmm.classes=$class_name \
        gmm.batch_size=null \
        gmm.n_samples_fit=$n_samples_fit \
        gmm.n_samples_generate=$n_samples_generate \
        gmm.save_dir=$save_dir/$class_name \
        gmm.stats_save_dir=$stats_save_dir \
        gmm.sample_idx=$sample_idx \
        rseed=$rseed
done