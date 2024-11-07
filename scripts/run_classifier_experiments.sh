# Set variable to main (parent) directory
main_dir=$(dirname "$(dirname "$0")")
data_dir=$WORK/vision_datasets

python scripts/train_classifier.py \
        experiment.data_dir=$data_dir/imagenette64 \
        classifier.max_allowed_samples_per_class=1000 \
        classifier.criterion=MSELoss \
        classifier.class_list='["english_springer", "french_horn"]' \
        classifier.num_epochs=40 \
        classifier.lr=1e-3 \
        classifier.train_split=0.8 \
        classifier.batch_size=64 \
        classifier.device=cuda:5 \
        classifier.n_runs=4 \
        classifier.verbose=false \
        "$@"
