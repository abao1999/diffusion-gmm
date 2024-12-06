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

python scripts/plot_spectra.py \
        --real_path None \
        --gmm_path results/gram_spectrum/${dataset_name}_gmm_gram_spectrum.npy \
        --diffusion_path results/gram_spectrum/${dataset_name}_diffusion_gram_spectrum.npy \
        --dataset_name $dataset_name \
        --save_dir final_plots/gram_spectrum \
        --save_name ${dataset_name}_all_spectra