# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

# python scripts/train_eval_classifier.py \
#         classifier_experiment.data_dir=$data_dir/cifar10 \
#         classifier_experiment.dataset_name=cifar10 \
#         classifier_experiment.batch_size=64 \
#         classifier_experiment.num_epochs=40 \
#         classifier_experiment.lr=1e-3 \
#         classifier_experiment.split_ratio=0.8 \
#         classifier_experiment.class_list=["dog", "cat"] \
#         classifier_experiment.device=cuda:0 \
        

# python scripts/train_eval_classifier.py \
#         classifier_experiment.data_dir=$data_dir/imagenette64 \
#         classifier_experiment.class_list='["english_springer", "french_horn"]' \
#         classifier_experiment.batch_size=64 \
#         classifier_experiment.num_epochs=40 \
#         classifier_experiment.lr=1e-3 \
#         classifier_experiment.split_ratio=0.8 \
#         classifier_experiment.device=cuda:0 \
#         classifier_experiment.num_samples=2048 \

python scripts/train_eval_classifier.py \
        classifier_experiment.data_dir=$data_dir/edm_imagenet64 \
        classifier_experiment.class_list='["english_springer", "french_horn"]' \
        classifier_experiment.batch_size=64 \
        classifier_experiment.num_epochs=50 \
        classifier_experiment.lr=1e-3 \
        classifier_experiment.split_ratio=0.8 \
        classifier_experiment.device=cuda:0 \
        classifier_experiment.num_samples=2048 \
