data_dir=$WORK/vision_datasets
results_save_dir=results/classifier_debug

# experiment
n_runs=1
n_props_train=1

# training
num_epochs=10
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
n_train_samples_per_class=1024
n_test_samples_per_class=1024

class_list=(
    "baseball"
    "cauliflower"
    "church"
    "coral_reef"
)

class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
echo "class_list: $class_list_json"

if [ ${#class_list[@]} -le 4 ]; then
    run_name=$(IFS=-; echo "${class_list[*]}")
else
    run_name="${#class_list[@]}_classes"
fi
echo $run_name

# run
datetime=$(date +%m-%d_%H-%M-%S)

dataset=edm_imagenet64_all
criterion=MSELoss

echo "dataset: $dataset"
echo "criterion: $criterion"

python scripts/train_classifier.py \
    classifier.train_data_dir=$data_dir/$dataset \
    classifier.train_split=$train_split \
    classifier.max_allowed_samples_per_class=$max_allowed_samples_per_class \
    classifier.n_train_samples_per_class=$n_train_samples_per_class \
    classifier.n_test_samples_per_class=$n_test_samples_per_class \
    classifier.class_list=${class_list_json} \
    classifier.n_runs=$n_runs \
    classifier.n_props_train=$n_props_train \
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
    classifier.save_dir=$results_save_dir/${model_class}/$run_name \
    classifier.save_name=${dataset}_${criterion}_${run_name}_${datetime}_debug \
    classifier.device=cuda:2 \
    classifier.verbose=true \
    rseed=99