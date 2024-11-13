# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

n_samples=4096
target_class="english_springer"

# Experiment for imagenet downsized to 64x64 (imagenet64)

# # compute gram spectrum for imagenet64 class
# python scripts/compute_gram_spectrum.py \
#         experiment.data_dir=$data_dir/imagenette64 \
#         experiment.batch_size=32 \
#         experiment.target_class=$target_class \
#         experiment.num_samples=$n_samples \
#         experiment.save_dir=results/gram_spectrum \
#         experiment.save_name=imagenet64_${target_class}_gram_spectrum.npy \

# # compute gram spectrum for imagenet64 class gmm samples
# python scripts/compute_gram_spectrum.py \
#         experiment.data_dir=$data_dir/gmm_imagenet64 \
#         experiment.load_npy=true \
#         experiment.batch_size=32 \
#         experiment.target_class=$target_class \
#         experiment.num_samples=$n_samples \
#         experiment.save_dir=results/gram_spectrum \
#         experiment.save_name=imagenet64_${target_class}_gmm_gram_spectrum.npy \

# # compute gram spectrum for imagenet64 class diffusion samples
# python scripts/compute_gram_spectrum.py \
#         experiment.data_dir=$data_dir/edm_imagenet64 \
#         experiment.batch_size=32 \
#         experiment.target_class=$target_class \
#         experiment.num_samples=$n_samples \
#         experiment.save_dir=results/gram_spectrum \
#         experiment.save_name=imagenet64_${target_class}_edm_gram_spectrum.npy \

# compute gram spectrum for imagenet64 class gmm samples
python scripts/compute_gram_spectrum.py \
        experiment.data_dir=$data_dir/gmm_edm_imagenet64_big \
        experiment.load_npy=true \
        experiment.batch_size=32 \
        experiment.target_class=$target_class \
        experiment.num_samples=$n_samples \
        experiment.save_dir=results/gram_spectrum \
        experiment.save_name=imagenet64_${target_class}_gmm_gram_spectrum_run3.npy \

# run2 was made with gmm_imagenette64

python scripts/plot_spectra.py \
        --real_path results/gram_spectrum/imagenet64_${target_class}_gram_spectrum.npy \
        --gmm_path results/gram_spectrum/imagenet64_${target_class}_gmm_gram_spectrum_run3.npy \
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
#         experiment.data_dir=$data_dir/cifar10 \
#         experiment.dataset_name=cifar10 \
#         experiment.batch_size=64 \
#         experiment.num_samples=$n_samples \
#         experiment.save_dir=results/gram_spectrum \
#         experiment.save_name=cifar10_gram_spectrum.npy \

# # compute gram spectrum for cifar10 all classes gmm samples
# python scripts/compute_gram_spectrum.py \
#         experiment.data_dir=$data_dir/gmm_cifar10 \
#         experiment.load_npy=true \
#         experiment.batch_size=64 \
#         experiment.num_samples=$n_samples \
#         experiment.save_dir=results/gram_spectrum \
#         experiment.save_name=cifar10_gmm_gram_spectrum.npy \

# # compute gram spectrum for cifar10 all classes diffusion samples
# python scripts/compute_gram_spectrum.py \
#         experiment.data_dir=$data_dir/diffusion_cifar10 \
#         experiment.batch_size=64 \
#         experiment.num_samples=$n_samples \
#         experiment.save_dir=results/gram_spectrum \
#         experiment.save_name=cifar10_diffusion_gram_spectrum.npy \

# python scripts/plot_spectra.py \
#         --real_path results/gram_spectrum/cifar10_gram_spectrum.npy \
#         --gmm_path results/gram_spectrum/cifar10_gmm_gram_spectrum.npy \
#         --diffusion_path results/gram_spectrum/cifar10_diffusion_gram_spectrum.npy 