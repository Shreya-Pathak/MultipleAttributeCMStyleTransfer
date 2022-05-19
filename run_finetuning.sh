#!/bin/bash

python main.py \
--do_train --do_eval --do_predict \
--source_lang='en' --target_lang='cm' \
--model_name_or_path='./models/few_shot/base' \
--output_dir='./models/few_shot/pretrained_vec/train_100' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='./data/few_shot_training/fewshot_100.tsv' \
--validation_file='./data/cmi_control_dev_vector.tsv' \
--test_file='./data/cmi_control_test_vector_oracle.tsv' \
--num_train_epochs=3.0 \
--learning_rate=5e-5 \
--eval_steps=5 \
--save_steps=5 \
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
--save_total_limit=2