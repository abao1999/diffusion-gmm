main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets
n_samples_fit=10240
n_samples_generate=1600
sample_idx=6400
dataset_name="edm_imagenet64_all"
save_dataset_name="gmm_edm_imagenet64_test"
save_dir=$data_dir/$save_dataset_name
rseed=81

class_list=("tench" "church" "english_springer" "french_horn")

for i in "${!class_list[@]}"; do
    class_name=${class_list[i]}
    # read -p "Fit GMM on $n_samples_fit samples from '$dataset_name' class $class_name and save $n_samples_generate samples to '$save_dataset_name'? (y/n): " confirmation
    # if [[ "$confirmation" != "y" ]]; then
    #     echo "Generation aborted."
    #     exit 1
    # fi
    python scripts/sample_gmm.py \
        gmm.n_components=1 \
        gmm.data_dir=$data_dir/$dataset_name \
        gmm.covariance_type=full \
        gmm.classes=$class_name \
        gmm.batch_size=null \
        gmm.n_samples_fit=$n_samples_fit \
        gmm.n_samples_generate=$n_samples_generate \
        gmm.save_dir=$save_dir/$class_name \
        gmm.sample_idx=$sample_idx \
        rseed=$rseed
done
