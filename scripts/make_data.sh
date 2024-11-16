# # Set variable to main (parent) directory
# main_dir=$(dirname "$(dirname "$0")")
# data_dir=$WORK/vision_datasets
# class_name="french_horn"
# n_samples_fit=4096
# n_samples_generate=4096
# # dataset_name="imagenette64"
# dataset_name="edm_imagenet64_big"

# python scripts/sample_gmm.py \
#         gmm.n_components=1 \
#         gmm.data_dir=$data_dir/$dataset_name \
#         gmm.covariance_type=full \
#         gmm.batch_size=32 \
#         gmm.target_class=$class_name \
#         gmm.n_samples_fit=$n_samples_fit \
#         gmm.n_samples_generate=$n_samples_generate \
#         gmm.save_dir=$data_dir/gmm_$dataset_name/$class_name \



# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets
n_samples_fit=5120
n_samples_generate=5120
split=""
dataset_name="edm_imagenet64_all"
save_dataset_name="gmm_${dataset_name}${split:+_$split}"
save_dir=$data_dir/$save_dataset_name
rseed=999

for class_name in "english_springer" "french_horn"; do
    read -p "Fit GMM on $n_samples_fit samples from '$dataset_name' class $class_name and save $n_samples_generate samples to '$save_dataset_name'? (y/n): " confirmation
    if [[ "$confirmation" != "y" ]]; then
        echo "Generation aborted."
        exit 1
    fi

    python scripts/sample_gmm.py \
            gmm.n_components=1 \
            gmm.data_dir=$data_dir/$dataset_name \
            gmm.covariance_type=full \
            gmm.batch_size=32 \
            gmm.target_class=$class_name \
            gmm.n_samples_fit=$n_samples_fit \
            gmm.n_samples_generate=$n_samples_generate \
            gmm.save_dir=$save_dir/$class_name \
            rseed=$rseed
done




# # sample edm for imagenet class. NOTE: this is done in edm repo
# torchrun --standalone --nproc_per_node=2 generate.py \
#         --outdir=$WORK/vision_datasets/edm_imagenet64/$target_class \
#         --class=$class_idx \
#         --seeds=0-1023 \
#         --batch=64 \
#         --steps=256 \
#         --S_churn=40 \
#         --S_min=0.05 \
#         --S_max=50 \
#         --S_noise=1.003 \
#         --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl

# # sample gmm for imagenet64 class
# python scripts/sample_gmm.py \
#         gmm.n_components=1 \
#         gmm.data_dir=$data_dir/imagenette64 \
#         gmm.covariance_type=full \
#         gmm.batch_size=32 \
#         gmm.target_class=$target_class \
#         gmm.n_samples_fit=$n_samples \
#         gmm.n_samples_generate=$n_samples \
#         gmm.save_dir=$data_dir/gmm_imagenet64/$target_class \