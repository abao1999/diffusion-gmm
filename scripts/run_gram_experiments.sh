# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

n_samples=2000
class_list=("english_springer")
dataset_name="Imagenet64"

for class in "${class_list[@]}"; do
        echo "Computing gram spectrum for class: $class"
        python scripts/compute_gram_spectrum.py \
                experiment.data_dir=$data_dir/edm_imagenet64_all \
                experiment.batch_size=64 \
                experiment.target_class=$class \
                experiment.num_samples=$n_samples \
                experiment.save_dir=results/gram_spectrum \
                experiment.save_name=${dataset_name}_${class}_edm_gram_spectrum.npy \

        python scripts/compute_gram_spectrum.py \
                experiment.data_dir=$data_dir/gmm_edm_imagenet64_all \
                experiment.load_npy=true \
                experiment.batch_size=64 \
                experiment.target_class=$class \
                experiment.num_samples=$n_samples \
                experiment.save_dir=results/gram_spectrum \
                experiment.save_name=${dataset_name}_${class}_gmm_gram_spectrum.npy \

done

        # python scripts/plot_spectra.py \
        #         --real_path None \
        #         --gmm_path results/gram_spectrum/${dataset_name}_${class}_gmm_gram_spectrum.npy \
        #         --diffusion_path results/gram_spectrum/${dataset_name}_${class}_edm_gram_spectrum.npy \
        #         --dataset_name $dataset_name \
        #         --class_name $class \
        #         --save_dir final_plots/gram_spectrum \
        #         --save_name ${dataset_name}_${class}_spectra


# python scripts/plot_spectra.py \
#         --real_path None \
#         --gmm_path results/gram_spectrum/${dataset_name}_gmm_gram_spectrum.npy \
#         --diffusion_path results/gram_spectrum/${dataset_name}_diffusion_gram_spectrum.npy \
#         --dataset_name $dataset_name \
#         --save_dir final_plots/gram_spectrum \
#         --save_name ${dataset_name}_all_spectra


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