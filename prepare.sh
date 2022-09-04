m=deberta-v3-large
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_A --output_dir tag_op/data/$m --encoder $m --mode dev
