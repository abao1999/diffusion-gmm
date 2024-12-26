data_dir=$WORK/vision_datasets
results_save_dir=results/classifier

# experiment
n_runs=2
n_props_train=6
reset_model_random_seed=true

# training
num_epochs=600
batch_size=128
lr=0.1
eval_epoch_interval=5
early_stopping_patience=60

# model and optimization
model_class=LinearMulticlassClassifier
criterion=MSELoss
optimizer_class=SGD
scheduler_class=CosineAnnealingWarmRestarts

# dataset
train_split=0.8
max_allowed_samples_per_class=8000
n_train_samples_per_class=2048
n_test_samples_per_class=1024

class_list=(
    "baseball"
    "cauliflower"
    "church"
    "coral_reef"
    "english_springer"
    "french_horn"
    "garbage_truck"
    "goldfinch"
    "kimono"
    "mountain_bike"
    "patas_monkey"
    "pizza"
    "planetarium"
    "polaroid"
    "racer"
    "salamandra"
    "tabby"
    "tench"
    "trimaran"
    "volcano"
)

class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
echo "class_list: $class_list_json"

if [ ${#class_list[@]} -le 4 ]; then
    run_name=$(IFS=-; echo "${class_list[*]}")
else
    run_name="${#class_list[@]}_classes"
fi
echo $run_name

dataset_list=("gmm_edm_imagenet64_all")

# run
datetime=$(date +%m-%d_%H-%M-%S)
rseeds=(312 1453 75 41 90)

for rseed in "${rseeds[@]}"; do
    echo "rseed: $rseed"
    for i in "${!dataset_list[@]}"; do
        dataset=${dataset_list[i]}
        device_idx=2
        echo "dataset: $dataset"
        echo "device_idx: $device_idx"
        python scripts/train_classifier.py \
            classifier.train_data_dir=$data_dir/$dataset \
            classifier.train_split=$train_split \
            classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
            classifier.n_train_samples_per_class=$n_train_samples_per_class \
            classifier.n_test_samples_per_class=$n_test_samples_per_class \
            classifier.class_list=${class_list_json} \
            classifier.n_runs=$n_runs \
            classifier.n_props_train=$n_props_train \
            classifier.reset_model_random_seed=$reset_model_random_seed \
            classifier.model.name=$model_class \
            classifier.criterion=$criterion \
            classifier.optimizer.name=$optimizer_class \
            classifier.scheduler.name=$scheduler_class \
            classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_0=$((num_epochs / 2)) \
            classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.eta_min=1e-2 \
            classifier.batch_size=$batch_size \
            classifier.lr=$lr \
            classifier.num_epochs=$num_epochs \
            classifier.eval_epoch_interval=$eval_epoch_interval \
            classifier.early_stopping_patience=$early_stopping_patience \
            classifier.save_dir=$results_save_dir/${model_class}_softmax_nobias/$run_name \
            classifier.save_name=${dataset}_${criterion}_${run_name}_${datetime}_seed${rseed} \
            classifier.device=cuda:${device_idx} \
            classifier.verbose=true \
            rseed=$rseed
    done
done


# data_dir=$WORK/vision_datasets
# results_save_dir=results/classifier_representations

# # experiment
# n_runs=1
# n_props_train=1
# reset_model_random_seed=true

# # training
# num_epochs=600
# batch_size=128
# lr=5e-2
# eval_epoch_interval=5
# early_stopping_patience=60

# # model and optimization
# model_class=LinearMulticlassClassifier
# criterion=MSELoss
# optimizer_class=SGD
# scheduler_class=CosineAnnealingWarmRestarts

# # dataset
# train_split=0.5
# max_allowed_samples_per_class=4096
# n_train_samples_per_class=1024
# n_test_samples_per_class=1024

# class_list=(
#     "baseball"
#     "cauliflower"
#     "church"
#     "coral_reef"
#     "english_springer"
#     "french_horn"
#     "garbage_truck"
#     "goldfinch"
#     "kimono"
#     "mountain_bike"
#     "patas_monkey"
#     "pizza"
#     "planetarium"
#     "polaroid"
#     "racer"
#     "salamandra"
#     "tabby"
#     "tench"
#     "trimaran"
#     "volcano"
# )

# class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
# echo "class_list: $class_list_json"

# if [ ${#class_list[@]} -le 4 ]; then
#     run_name=$(IFS=-; echo "${class_list[*]}")
# else
#     run_name="${#class_list[@]}_classes"
# fi
# echo $run_name

# dataset_list=("gmm_representations" "representations")

# # run
# datetime=$(date +%m-%d_%H-%M-%S)
# rseeds=(312 1453 75 41 90)

# for rseed in "${rseeds[@]}"; do
#     echo "rseed: $rseed"
#     for i in "${!dataset_list[@]}"; do
#         dataset=${dataset_list[i]}
#         device_idx=2
#         echo "dataset: $dataset"
#         echo "device_idx: $device_idx"
#         python scripts/train_classifier.py \
#             classifier.train_data_dir=$data_dir/$dataset \
#             classifier.train_split=$train_split \
#             classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
#             classifier.n_train_samples_per_class=$n_train_samples_per_class \
#             classifier.n_test_samples_per_class=$n_test_samples_per_class \
#             classifier.class_list=${class_list_json} \
#             classifier.n_runs=$n_runs \
#             classifier.n_props_train=$n_props_train \
#             classifier.reset_model_random_seed=$reset_model_random_seed \
#             classifier.model.name=$model_class \
#             classifier.criterion=$criterion \
#             classifier.optimizer.name=$optimizer_class \
#             classifier.scheduler.name=$scheduler_class \
#             classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_0=$((num_epochs / 2)) \
#             classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.eta_min=5e-3 \
#             classifier.batch_size=$batch_size \
#             classifier.lr=$lr \
#             classifier.num_epochs=$num_epochs \
#             classifier.eval_epoch_interval=$eval_epoch_interval \
#             classifier.early_stopping_patience=$early_stopping_patience \
#             classifier.save_dir=$results_save_dir/$model_class/$run_name \
#             classifier.save_name=${dataset}_${criterion}_${run_name}_${datetime}_seed${rseed} \
#             classifier.device=cuda:${device_idx} \
#             classifier.verbose=true \
#             rseed=$rseed
#     done
# done