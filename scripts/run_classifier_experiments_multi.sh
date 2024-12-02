# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

train_split=0.8 # set to 1.0 when using separate folder for test set
n_runs=5
n_props_train=6
reset_model_random_seed=false
save_dir=results/classifier
num_epochs=500
max_allowed_samples_per_class=8000
max_allowed_samples_per_class_test=0
n_train_samples_per_class=2048
n_test_samples_per_class=1024
batch_size=128
lr=1e-1 # 3e-4
train_augmentations=None
resample_train_subset=true
resample_test_subset=true
eval_epoch_interval=10
early_stopping_patience=100
model_save_dir=$WORK/vision_datasets/checkpoints
model_save_dir=null

model_class=LinearMulticlassClassifier
criterion=MSELoss
optimizer_class=SGD

scheduler_class=CosineAnnealingLR
rseed=1000
verbose=false

datetime=$(date +%m-%d_%H-%M-%S)

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
echo $class_list_json

if [ ${#class_list[@]} -le 4 ]; then
    run_name=$(IFS=-; echo "${class_list[*]}")
else
    run_name="${#class_list[@]}_classes"
fi
echo $run_name

dataset_list=("gmm_edm_imagenet64" "edm_imagenet64")

for i in "${!dataset_list[@]}"; do
    dataset=${dataset_list[i]}
    train_dataset=${dataset}_all
    device_idx=2
    echo $train_dataset
    echo $device_idx
    python scripts/train_classifier.py \
        classifier.train_data_dir=$data_dir/$train_dataset \
        classifier.test_data_dir=null \
        classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
        classifier.max_allowed_samples_per_class_test=$max_allowed_samples_per_class_test \
        classifier.n_train_samples_per_class=$n_train_samples_per_class \
        classifier.n_test_samples_per_class=$n_test_samples_per_class \
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
        classifier.eval_epoch_interval=$eval_epoch_interval \
        classifier.early_stopping_patience=$early_stopping_patience \
        classifier.model_save_dir=$model_save_dir \
        classifier.save_dir=$save_dir \
        classifier.save_name=${dataset}_bs${batch_size}_${criterion}_${run_name}_${datetime} \
        classifier.device=cuda:${device_idx} \
        classifier.optimizer.name=$optimizer_class \
        classifier.scheduler.name=$scheduler_class \
        classifier.scheduler.CosineAnnealingLR_kwargs.T_max=$num_epochs \
        classifier.scheduler.CosineAnnealingLR_kwargs.eta_min=1e-2 \
        classifier.verbose=$verbose \
        rseed=$rseed
done