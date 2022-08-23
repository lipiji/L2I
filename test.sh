m=roberta
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/data/$m --test_data_dir tag_op/data/A --save_dir tag_op/model_L2I/$m/test --eval_batch_size 32 --model_path tag_op/model_L2I/$m --encoder $m
#--roberta_model path_to_roberta_model

