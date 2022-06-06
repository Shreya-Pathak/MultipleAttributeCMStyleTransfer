#!/bin/bash

python run_fudge.py \
--input_filename='data/fudge/test_subset.tsv' \
--output_directory='results/fudge/translate_pretrained' \
--path_to_cmgen_model='models/trans_cmgen_masked' \
--predictor_name='base_en' \
--beam_width=4

# python run_fudge.py \
# --input_filename='data/fudge/treebank_test_current.txt' \
# --output_directory='results/fudge/translate_pretrained' \
# --path_to_cmgen_model='models/translate_pretrained_cmgen' \
# --predictor_name='trained_hi' \
# --beam_width=4 \
# --make_formal
