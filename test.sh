m=roberta
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --input_path ./dataset_A --output_dir tag_op/data/A --encoder $m --mode dev --data_format tatqa_and_hqa_dataset_{}.json
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/$m --test_data_dir tag_op/data/A --save_dir tag_op/model_L2I/$m/test --eval_batch_size 32 --model_path tag_op/model_L2I/$m --encoder $m
#--roberta_model path_to_roberta_model

