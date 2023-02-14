#!/bin/bash

python ../generate_lm_output.py -j generate_lm_output_gpt-j-6b.json
python ../generate_lm_output.py -j generate_lm_output_bloom-7b1.json
python ../generate_lm_output.py -j generate_lm_output_opt-6.7b.json