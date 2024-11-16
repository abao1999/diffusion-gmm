# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

train_split=1.0 # using separate folder for test set
n_runs=1
n_props_train=1
reset_model_random_seed=true
save_dir=results/classifier
num_epochs=400
max_allowed_samples_per_class=4096
batch_size=64
lr=1e-3  # 3e-4
# model_class=TwoLayerMulticlassClassifier
model_class=LinearBinaryClassifier
# criterion=CrossEntropyLoss
criterion=MSELoss
optimizer_class=SGD
# scheduler_class=null
scheduler_class=CosineAnnealingWarmRestarts
rseed=123
use_augmentations=false
verbose=true

dataset_list=("gmm_edm_imagenet64" "edm_imagenet64")
for i in "${!dataset_list[@]}"; do
    dataset=${dataset_list[i]}
    train_dataset=${dataset}_train
    test_dataset=${dataset}_test
    # test_dataset=edm_imagenet64_big_test
    # test_dataset=imagenette64
    device_idx=6
    echo $train_dataset
    echo $test_dataset
    echo $device_idx
    python scripts/train_classifier.py \
        classifier.train_data_dir=$data_dir/$train_dataset \
        classifier.test_data_dir=$data_dir/$test_dataset \
        classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
        classifier.model.name=$model_class \
        classifier.criterion=$criterion \
        classifier.class_list='["english_springer", "french_horn"]' \
        classifier.num_epochs=$num_epochs \
        classifier.lr=$lr \
        classifier.train_split=$train_split \
        classifier.batch_size=$batch_size \
        classifier.n_runs=$n_runs \
        classifier.n_props_train=$n_props_train \
        classifier.use_augmentations=$use_augmentations \
        classifier.reset_model_random_seed=$reset_model_random_seed \
        classifier.save_dir=$save_dir \
        classifier.save_name=${dataset}_bs${batch_size}_${criterion}_english-springer_french-horn \
        classifier.device=cuda:${device_idx} \
        classifier.optimizer.name=$optimizer_class \
        classifier.scheduler.name=$scheduler_class \
        classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_0=$((num_epochs / 2)) \
        classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_mult=1 \
        classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.eta_min=1e-5 \
        rseed=$rseed \
        classifier.verbose=$verbose \
        # "$@" &
done


        # classifier.scheduler.CosineAnnealingLR_kwargs.T_max=$num_epochs \
        # classifier.scheduler.CosineAnnealingLR_kwargs.eta_min=1e-5 \