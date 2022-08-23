m=xlm-roberta-large
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_extra_field --output_dir tag_op/data/$m --encoder $m  --mode dev
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_extra_field --output_dir tag_op/data/$m --encoder $m  --mode train
