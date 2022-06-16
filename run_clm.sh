#!/bin/bash

python run_clm.py \
--model_name_or_path="xlm-roberta-base" \
--train_file="./data/lm_training/tcs_downsampled_top_200k.txt" \
--validation_file="./data/lm_training/validation.txt" \
--test_file="./data/lm_training/test.txt" \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--gradient_accumulation_steps=2 \
--load_best_model_at_end \
--metric_for_best_model='perplexity' \
--eval_steps=10 \
--save_steps=10 \
--do_train \
--do_eval \
--do_predict \
--output_dir=./models/lm_ishan \
--evaluation_strategy='steps' \
--save_strategy='steps' \
--max_train_samples=100 \
--max_eval_samples=10 \
--max_predict_samples=10