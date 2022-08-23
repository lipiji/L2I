m=deberta-v2-xxlarge
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/trainer.py \
  --data_dir tag_op/data/$m \
  --save_dir tag_op/model_L2I/$m \
  --batch_size 5 \
  --eval_batch_size 5 \
  --max_epoch 100 \
  --warmup 0.06 \
  --optimizer adam \
  --learning_rate 5e-4  \
  --weight_decay 5e-5 \
  --seed 123 \
  --gradient_accumulation_steps 5 \
  --bert_learning_rate 1.5e-5 \
  --bert_weight_decay 0.01 \
  --log_per_updates 100 \
  --eps 1e-6  \
  --encoder $m \
  --test_data_dir tag_op/data/$m/ \
  --cross_attn_layer 0
#  --roberta_model roberta_model

