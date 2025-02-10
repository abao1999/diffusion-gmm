data_dir=$WORK/vision_datasets
results_save_dir=results/classifier

# experiment
n_runs=1
n_props_train=1
reset_model_random_seed=true

# training
num_epochs=600
batch_size=64
lr=0.1

# model and optimization
model_class=LinearMulticlassClassifier
criterion=MSELoss
optimizer_class=SGD
scheduler_class=CosineAnnealingWarmRestarts

# dataset
train_split=0.5
max_allowed_samples_per_class=4096
n_train_samples_per_class=1024
n_test_samples_per_class=1024

class_list=(
    "church"
    "tench"
    "english_springer"
    "french_horn"
)

class_list_json=$(printf '%s\n' "${class_list[@]}" | jq -R . | jq -s -c .)
echo "class_list: $class_list_json"

if [ ${#class_list[@]} -le 4 ]; then
    run_name=$(IFS=-; echo "${class_list[*]}")
else
    run_name="${#class_list[@]}_classes"
fi
echo $run_name

dataset_list=("edm2_representations" "gmm_edm2_representations")

# run
datetime=$(date +%m-%d_%H-%M-%S)
rseeds=(12 27 124 100 46)

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
            classifier.model.use_bias=true \
            classifier.model.output_logit=false \
            classifier.criterion=$criterion \
            classifier.optimizer.name=$optimizer_class \
            classifier.scheduler.name=$scheduler_class \
            classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.T_0=$((num_epochs / 2)) \
            classifier.scheduler.CosineAnnealingWarmRestarts_kwargs.eta_min=2e-5 \
            classifier.batch_size=$batch_size \
            classifier.lr=$lr \
            classifier.num_epochs=$num_epochs \
            classifier.save_dir=$results_save_dir/${model_class}/$run_name \
            classifier.save_name=${dataset}_${criterion}_${run_name}_${datetime}_seed${rseed} \
            classifier.device=cuda:${device_idx} \
            classifier.verbose=true \
            rseed=$rseed
    done
done
