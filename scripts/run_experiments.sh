# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

n_samples=1024
target_class="french_horn"

# Experiment for imagenet downsized to 64x64 (imagenet64)

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

# sample gmm for imagenet64 class
python scripts/sample_gmm.py \
        gmm.n_components=1 \
        gmm.data_dir=$data_dir/imagenette64 \
        gmm.covariance_type=full \
        gmm.batch_size=32 \
        gmm.target_class=$target_class \
        gmm.n_samples_fit=$n_samples \
        gmm.n_samples_generate=$n_samples \
        gmm.save_dir=$data_dir/gmm_imagenet64/$target_class \

# compute gram spectrum for imagenet64 class
python scripts/compute_gram_spectrum.py \
        gram_experiment.data_dir=$data_dir/imagenette64 \
        gram_experiment.batch_size=32 \
        gram_experiment.target_class=$target_class \
        gram_experiment.num_samples=$n_samples \
        gram_experiment.save_dir=results/gram_spectrum \
        gram_experiment.save_name=imagenet64_${target_class}_gram_spectrum.npy \

# compute gram spectrum for imagenet64 class gmm samples
python scripts/compute_gram_spectrum.py \
        gram_experiment.data_dir=$data_dir/gmm_imagenet64 \
        gram_experiment.load_npy=true \
        gram_experiment.batch_size=32 \
        gram_experiment.target_class=$target_class \
        gram_experiment.num_samples=$n_samples \
        gram_experiment.save_dir=results/gram_spectrum \
        gram_experiment.save_name=imagenet64_${target_class}_gmm_gram_spectrum.npy \

# compute gram spectrum for imagenet64 class diffusion samples
python scripts/compute_gram_spectrum.py \
        gram_experiment.data_dir=$data_dir/edm_imagenet64 \
        gram_experiment.batch_size=32 \
        gram_experiment.target_class=$target_class \
        gram_experiment.num_samples=$n_samples \
        gram_experiment.save_dir=results/gram_spectrum \
        gram_experiment.save_name=imagenet64_${target_class}_edm_gram_spectrum.npy \

python scripts/plot_spectra.py \
        --real_path results/gram_spectrum/imagenet64_${target_class}_gram_spectrum.npy \
        --gmm_path results/gram_spectrum/imagenet64_${target_class}_gmm_gram_spectrum.npy \
        --diffusion_path results/gram_spectrum/imagenet64_${target_class}_edm_gram_spectrum.npy 

# Experiment for cifar10
# n_samples=2048

# # sample diffusion for cifar10 all classes
# python scripts/sample_diffusion.py \
#         diffusion.steps=200\
#         diffusion.n_samples=$n_samples \
#         diffusion.save_dir=$data_dir/diffusion_cifar10/unknown \
#         diffusion.verbose=true \

# # sample gmm for cifar10 all classes
# python scripts/sample_gmm.py \
#         gmm.n_components=10 \
#         gmm.data_dir=$data_dir/cifar10 \
#         gmm.dataset_name=cifar10 \
#         gmm.covariance_type=full \
#         gmm.batch_size=32 \
#         gmm.n_samples_fit=2048 \
#         gmm.n_samples_generate=$n_samples \
#         gmm.save_dir=$data_dir/gmm_cifar10/unknown \

# # compute gram spectrum for cifar10 all classes
# python scripts/compute_gram_spectrum.py \
#         gram_experiment.data_dir=$data_dir/cifar10 \
#         gram_experiment.dataset_name=cifar10 \
#         gram_experiment.batch_size=64 \
#         gram_experiment.num_samples=$n_samples \
#         gram_experiment.save_dir=results/gram_spectrum \
#         gram_experiment.save_name=cifar10_gram_spectrum.npy \

# # compute gram spectrum for cifar10 all classes gmm samples
# python scripts/compute_gram_spectrum.py \
#         gram_experiment.data_dir=$data_dir/gmm_cifar10 \
#         gram_experiment.load_npy=true \
#         gram_experiment.batch_size=64 \
#         gram_experiment.num_samples=$n_samples \
#         gram_experiment.save_dir=results/gram_spectrum \
#         gram_experiment.save_name=cifar10_gmm_gram_spectrum.npy \

# # compute gram spectrum for cifar10 all classes diffusion samples
# python scripts/compute_gram_spectrum.py \
#         gram_experiment.data_dir=$data_dir/diffusion_cifar10 \
#         gram_experiment.batch_size=64 \
#         gram_experiment.num_samples=$n_samples \
#         gram_experiment.save_dir=results/gram_spectrum \
#         gram_experiment.save_name=cifar10_diffusion_gram_spectrum.npy \

# python scripts/plot_spectra.py \
#         --real_path results/gram_spectrum/cifar10_gram_spectrum.npy \
#         --gmm_path results/gram_spectrum/cifar10_gmm_gram_spectrum.npy \
#         --diffusion_path results/gram_spectrum/cifar10_diffusion_gram_spectrum.npy 