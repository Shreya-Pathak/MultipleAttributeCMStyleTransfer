#!/bin/bash

python main.py \
--do_predict \
--model_name_or_path='./models/mt5_hd_ft_cmi_vector/checkpoint-70000' \
--source_lang='en' --target_lang='cm' \
--output_dir='./models/mt5_hd_ft_cmi_vector' \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--predict_with_generate \
--test_file='./data/cmi_control_test_vector_oracle.tsv' \
--validation_file='./data/cmi_control_dev_vector.tsv' \
--generation_num_beams=1 \
--generation_max_length=128 \
--max_source_length=128 \
--max_target_length=128