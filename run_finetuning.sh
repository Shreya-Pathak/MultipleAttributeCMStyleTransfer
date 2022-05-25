#!/bin/bash

python main.py \
--do_train --do_eval \
--source_lang='en' --target_lang='cm' \
--output_dir='models/mt5_hd_ft_cmi_vector_new' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--overwrite_output_dir=False \
--predict_with_generate \
--train_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_train_vector.tsv' \
--validation_file='/home/shreyapathak9515/controllable_codemixing/cm_data/cmi_control_dev_vector.tsv' \
--load_best_model_at_end \
--metric_for_best_model='cmi_spi_bleu_hm' \
--num_train_epochs=30.0 \
--learning_rate=5e-4 \
--eval_steps=1000 \
--save_steps=1000 \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--lr_scheduler_type='constant' \
--generation_num_beams=1 \
--generation_max_length=128 \
--optim='adafactor' \
--max_source_length=128 \
--max_target_length=128 \
--save_total_limit=1
