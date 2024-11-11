# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

n_runs=20
n_props_train=4
reset_model_random_seed=true
save_dir=results/classifier
num_epochs=200
max_allowed_samples_per_class=1024
train_split=0.8
batch_size=64
lr=3e-4
criterion=MSELoss
optimizer_class=SGD
scheduler_class=null
rseed=10
verbose=false


dataset_list=("imagenette64" "gmm_imagenet64" "edm_imagenet64")
for i in "${!dataset_list[@]}"; do
    dataset=${dataset_list[i]}
    device_idx=$((6 - i))
    echo $dataset
    echo $device_idx
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python scripts/train_classifier.py \
        experiment.data_dir=$data_dir/$dataset \
        classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
        classifier.criterion=$criterion \
        classifier.class_list='["english_springer", "french_horn"]' \
        classifier.num_epochs=$num_epochs \
        classifier.lr=$lr \
        classifier.train_split=$train_split \
        classifier.batch_size=$batch_size \
        classifier.n_runs=$n_runs \
        classifier.n_props_train=$n_props_train \
        classifier.reset_model_random_seed=$reset_model_random_seed \
        classifier.save_dir=$save_dir \
        classifier.save_name=${dataset}_english-springer_french-horn \
        classifier.device=cuda:${device_idx} \
        classifier.optimizer.method=$optimizer_class \
        classifier.scheduler.method=$scheduler_class \
        rseed=$rseed \
        classifier.verbose=$verbose \
        # "$@" &
done