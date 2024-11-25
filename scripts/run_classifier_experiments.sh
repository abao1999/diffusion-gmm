# # Set variable to main (parent) directory
# main_dir=$(dirname "$(dirname "$0")")
# data_dir=$WORK/vision_datasets

# train_split=1.0 # using separate folder for test set
# n_runs=4
# n_props_train=6
# reset_model_random_seed=true
# save_dir=results/classifier
# num_epochs=500
# max_allowed_samples_per_class=5120
# max_allowed_samples_per_class_test=1280
# batch_size=128
# lr=1e-2  # 3e-4
# train_augmentations=None
# resample_train_subset=true
# resample_test_subset=true

# model_class=LinearBinaryClassifier
# criterion=MSELoss
# optimizer_class=SGD

# # scheduler_class=CosineAnnealingWarmRestarts
# scheduler_class=CosineAnnealingLR
# rseed=301
# verbose=true

# datetime=$(date +%m-%d_%H-%M-%S)

# class_list=("church" "tench")
# # class_list=("english_springer" "french_horn")
# class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)

# echo $class_list_json

# dataset_list=("edm_imagenet64" "gmm_edm_imagenet64")

# for i in "${!dataset_list[@]}"; do
#     dataset=${dataset_list[i]}
#     train_dataset=${dataset}_train
#     test_dataset=${dataset}_test
#     device_idx=2
#     echo $train_dataset
#     echo $test_dataset
#     echo $device_idx
#     python scripts/train_classifier.py \
#         classifier.train_data_dir=$data_dir/$train_dataset \
#         classifier.test_data_dir=$data_dir/$test_dataset \
#         classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
#         classifier.max_allowed_samples_per_class_test=$max_allowed_samples_per_class_test \
#         classifier.model.name=$model_class \
#         classifier.criterion=$criterion \
#         classifier.class_list=${class_list_json} \
#         classifier.num_epochs=$num_epochs \
#         classifier.lr=$lr \
#         classifier.train_split=$train_split \
#         classifier.batch_size=$batch_size \
#         classifier.n_runs=$n_runs \
#         classifier.n_props_train=$n_props_train \
#         classifier.train_augmentations=$train_augmentations \
#         classifier.resample_train_subset=$resample_train_subset \
#         classifier.resample_test_subset=$resample_test_subset \
#         classifier.reset_model_random_seed=$reset_model_random_seed \
#         classifier.save_dir=$save_dir \
#         classifier.save_name=${dataset}_bs${batch_size}_${criterion}_${class_list[0]}_${class_list[1]}_${datetime} \
#         classifier.device=cuda:${device_idx} \
#         classifier.optimizer.name=$optimizer_class \
#         classifier.scheduler.name=$scheduler_class \
#         classifier.scheduler.CosineAnnealingLR_kwargs.T_max=$num_epochs \
#         classifier.scheduler.CosineAnnealingLR_kwargs.eta_min=1e-4 \
#         rseed=$rseed \
#         classifier.verbose=$verbose
# done


    # classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_0=$((num_epochs / 2)) \
    # classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_mult=1 \
    # classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.eta_min=1e-3 \



# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

train_split=1.0 # using separate folder for test set
n_runs=4
n_props_train=1
reset_model_random_seed=true
save_dir=results/classifier
num_epochs=500
max_allowed_samples_per_class=3072
max_allowed_samples_per_class_test=768
batch_size=128
lr=1e-2  # 3e-4
train_augmentations=None
resample_train_subset=true
resample_test_subset=true
early_stopping_patience=50

model_class=LinearMulticlassClassifier
criterion=MSELoss
optimizer_class=SGD

# scheduler_class=CosineAnnealingWarmRestarts
scheduler_class=CosineAnnealingLR
rseed=133
verbose=true

datetime=$(date +%m-%d_%H-%M-%S)

class_list=("english_springer" "french_horn" "church" "tench")
class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)

echo $class_list_json

dataset_list=("edm_imagenet64" "gmm_edm_imagenet64")

for i in "${!dataset_list[@]}"; do
    dataset=${dataset_list[i]}
    train_dataset=${dataset}_train
    test_dataset=${dataset}_test
    device_idx=2
    echo $train_dataset
    echo $test_dataset
    echo $device_idx
    python scripts/train_classifier.py \
        classifier.train_data_dir=$data_dir/$train_dataset \
        classifier.test_data_dir=$data_dir/$test_dataset \
        classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
        classifier.max_allowed_samples_per_class_test=$max_allowed_samples_per_class_test \
        classifier.model.name=$model_class \
        classifier.criterion=$criterion \
        classifier.class_list=${class_list_json} \
        classifier.num_epochs=$num_epochs \
        classifier.lr=$lr \
        classifier.train_split=$train_split \
        classifier.batch_size=$batch_size \
        classifier.n_runs=$n_runs \
        classifier.n_props_train=$n_props_train \
        classifier.train_augmentations=$train_augmentations \
        classifier.resample_train_subset=$resample_train_subset \
        classifier.resample_test_subset=$resample_test_subset \
        classifier.reset_model_random_seed=$reset_model_random_seed \
        classifier.early_stopping_patience=$early_stopping_patience \
        classifier.save_dir=$save_dir \
        classifier.save_name=${dataset}_bs${batch_size}_${criterion}_${class_list[0]}_${class_list[1]}_${class_list[2]}_${class_list[3]}_${datetime} \
        classifier.device=cuda:${device_idx} \
        classifier.optimizer.name=$optimizer_class \
        classifier.scheduler.name=$scheduler_class \
        classifier.scheduler.CosineAnnealingLR_kwargs.T_max=$num_epochs \
        classifier.scheduler.CosineAnnealingLR_kwargs.eta_min=1e-4 \
        rseed=$rseed \
        classifier.verbose=$verbose
done
