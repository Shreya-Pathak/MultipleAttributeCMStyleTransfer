#!/bin/bash

python run_fudge.py \
--input_filename='data/fudge/test_subset_1000.tsv' \
--output_directory='results/fudge/translation_multitask_cmi_vec_formal' \
--path_to_cmgen_model='models/translation_multitask_cmi_vec' \
--predictor_name='train_en' \
--beam_width=4 \
--make_formal

python run_fudge.py \
--input_filename='data/fudge/test_subset_1000.tsv' \
--output_directory='results/fudge/translation_multitask_cmi_vec_formal' \
--path_to_cmgen_model='models/translation_multitask_cmi_vec' \
--predictor_name='train_hi' \
--beam_width=4 \
--make_formal

python run_fudge.py \
--input_filename='data/fudge/test_subset_1000.tsv' \
--output_directory='results/fudge/translation_multitask_cmi_vec_informal' \
--path_to_cmgen_model='models/translation_multitask_cmi_vec' \
--predictor_name='train_hi_dev_cs' \
--beam_width=4