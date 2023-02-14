#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python generate_lm_output_decoding.py -j ./generate_lm_output_vC/generate_lm_output_decoding_opt-6.7b.json
CUDA_VISIBLE_DEVICES=0 python generate_lm_output_decoding.py -j ./generate_lm_output_vC/generate_lm_output_decoding_opt-13b.json
CUDA_VISIBLE_DEVICES=0,1 python generate_lm_output_decoding.py -j ./generate_lm_output_vC/generate_lm_output_decoding_opt-30b.json
