#!/bin/bash

for train_set_size in 10 20 30 40 50 60 70 80 90 100
do
	train_samples=$((train_set_size*22982/100))
	echo "Running for train set size : ${train_samples}"
	python main.py \
	--do_train --do_eval --do_predict \
	--source_lang='en' --target_lang='cm' \
	--model_name_or_path='./models/few_shot/base' \
	--output_dir="./models/few_shot/pretrained_vec_last2_lr5e-4/train_${train_set_size}" \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--gradient_accumulation_steps=2 \
	--overwrite_output_dir=False \
	--predict_with_generate \
	--train_file="./data/cmi_control_train_vector.tsv" \
	--validation_file='./data/cmi_control_dev_vector.tsv' \
	--test_file='./data/cmi_control_test_vector_oracle.tsv' \
	--num_train_epochs=10.0 \
	--learning_rate=5e-4 \
	--eval_steps=100 \
	--save_steps=100 \
	--evaluation_strategy='steps' \
	--save_strategy='steps' \
	--lr_scheduler_type='constant' \
	--generation_num_beams=1 \
	--generation_max_length=128 \
	--optim='adafactor' \
	--max_source_length=128 \
	--max_target_length=128 \
	--load_best_model_at_end \
	--metric_for_best_model='cmi_acc_bleu_hm' \
	--save_total_limit=1 \
	--max_train_samples=${train_samples}
done