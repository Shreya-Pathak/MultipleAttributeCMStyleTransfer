#!/bin/bash
for n in {3..5};
do	
	python run_clm.py \
	--model_name_or_path="xlm-roberta-base" \
	--train_file="./data/lm_training/tcs_data/tcs_200k_$n.txt" \
	--validation_file="./data/lm_training/validation.txt" \
	--test_file="./data/lm_training/test.txt" \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--gradient_accumulation_steps=2 \
	--load_best_model_at_end \
	--metric_for_best_model='loss' \
	--eval_steps=200 \
	--save_steps=200 \
	--learning_rate=5e-6 \
	--do_train \
	--do_eval \
	--do_predict \
	--output_dir=./models/lm_tcs/tcs_200k_$n \
	--evaluation_strategy='steps' \
	--save_strategy='steps' \
	--num_train_epochs=10.0 \
	--save_total_limit=1 \
	--block_size=128
done
