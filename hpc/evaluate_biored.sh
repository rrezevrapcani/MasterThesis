#!/bin/bash

python scripts/eval/run_biored_eval.py --exp_option 'to_pubtator' \
  --in_test_pubtator_file datasets/biored/BioRED/Test.PubTator \
  --in_test_tsv_file datasets/biored/processed/test.tsv \
  --in_pred_rel_tsv_file outputs/biored/predictions.tsv \
  --in_pred_novelty_tsv_file outputs/biored/dummy_novelty_test_results.tsv \
  --out_pred_pubtator_file outputs/biored/biored_pred_mul.txt

python scripts/eval/run_biored_eval.py --exp_option 'biored_eval' \
  --in_gold_pubtator_file datasets/biored/BioRED/Test.PubTator \
  --in_pred_pubtator_file outputs/biored/biored_pred_mul.txt