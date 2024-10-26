# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets


n_samples=2048

# sample diffusion for cifar10 all classes
python scripts/sample_diffusion.py \
        diffusion.steps=200\
        diffusion.n_samples=$n_samples \
        diffusion.save_dir=$data_dir/diffusion_cifar10/unknown \
        diffusion.verbose=true \

# sample gmm for cifar10 all classes
python scripts/sample_gmm.py \
        gmm.n_components=10 \
        gmm.data_dir=$data_dir/cifar10 \
        gmm.dataset_name=cifar10 \
        gmm.covariance_type=full \
        gmm.batch_size=32 \
        gmm.n_samples_fit=2048 \
        gmm.n_samples_generate=$n_samples \
        gmm.save_dir=$data_dir/gmm_cifar10/unknown \

# compute gram spectrum for cifar10 all classes
python scripts/compute_gram_spectrum.py \
        gs_exp.data_dir=$data_dir/cifar10 \
        gs_exp.dataset_name=cifar10 \
        gs_exp.batch_size=64 \
        gs_exp.num_samples=$n_samples \
        gs_exp.save_dir=results/gram_spectrum \
        gs_exp.save_name=cifar10_gram_spectrum.npy \

# compute gram spectrum for cifar10 all classes gmm samples
python scripts/compute_gram_spectrum.py \
        gs_exp.data_dir=$data_dir/gmm_cifar10 \
        gs_exp.load_npy=true \
        gs_exp.batch_size=64 \
        gs_exp.num_samples=$n_samples \
        gs_exp.save_dir=results/gram_spectrum \
        gs_exp.save_name=cifar10_gmm_gram_spectrum.npy \

# compute gram spectrum for cifar10 all classes diffusion samples
python scripts/compute_gram_spectrum.py \
        gs_exp.data_dir=$data_dir/diffusion_cifar10 \
        gs_exp.batch_size=64 \
        gs_exp.num_samples=$n_samples \
        gs_exp.save_dir=results/gram_spectrum \
        gs_exp.save_name=cifar10_diffusion_gram_spectrum.npy \

python scripts/plot_spectra.py \
        --real_path results/gram_spectrum/cifar10_gram_spectrum.npy \
        --gmm_path results/gram_spectrum/cifar10_gmm_gram_spectrum.npy \
        --diffusion_path results/gram_spectrum/cifar10_diffusion_gram_spectrum.npy 