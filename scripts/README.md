# Scripts

Put your standalone scripts (Python and bash) as well as jupyter notebooks here.

# Stages of processing:
1. `prepare_dataset_to_prompts/prepare_dataset_to_prompts.py`: This script takes as input the csv file and split it in two or three parts. Then, it stores them as csv. Additionally, it stores them with prompt format. 

1. `generate_input_prompts/generate_input_prompts.py`: This script takes as input the train or validation messages and generates prompts with specific number of example and format taken randomly from the input file.

1. `generate_lm_output/generate_lm_output.py`: It loads the model and gets generated sentences for different prompt version

1. `generate_lm_output/generate_lm_output_decoding.py`: It loads the model and gets generated sentences for different decoding versions.

1. `get_sentences_from_output/get_sentences_from_output_vB_v2.py`: It splits the string obtained in order to obtain the generated messages for prompt selection

1. `get_sentences_from_output/get_sentences_from_output_vC_v2.py`: It splits the string obtained in order to obtained the generated messages for decoding selection.

1. '`get_statistics_messages_generated/get_statistics_messages_generated.py`: it produces two files: 
        * `statistics_summary.csv`: average number of messages per model/version
        * `statistics_index_prompt_level.csv`: number of messages per prompt

1. `postprocess_on_selfBLEU/postprocess_on_selfBLEU.py`: it obtains the BLEU-4 value

1. `discard_on_selfBLEU/discard_on_selfBLEU.py`: it discards messages based on threshold for selfBLEU. 

1. `discard_on_criteria/discard_on_criteria.py`: it discards messages based on certain criteria. For our case: presence of app or similar word, underscore(_) and message length less than 6.

1. `calculate_perplexity/calculate_perplexity.py`: it allows to calculate perplexity according to a specific language model

1. `join_results_perplexity/join_results_perplexity.py`: it joins the perplexities values from different models.

1. `join_all_sentences/join_all_sentences_v2.py`: it joins all final sentences through different combinations.

1. `join_all_sentences/join_all_sentences_v2_vC.py`: it joins all final sentences through different combinations and the original messages. The difference with the previous function is the number of outputs. It outputs 2 csv files: one with type original and the other with type: train or validation according to the belonging of the original message.

1. `process_LIWC_results/process_LIWC_results.py`" it calculates the mean and std error for selected LIWC metrics.

TODO:
add GLMM and EMMS for prompt and decoding 

1. 
Commands used:
```bash
python prepare_dataset_to_prompts -j prepare_dataset_to_prompts_v1.json
python generate_input_prompts.py -j generate_input_prompts_vB/generate_input_prompts_v1.json
python generate_input_prompts.py -j generate_input_prompts_vB/generate_input_prompts_v2.json
python generate_input_prompts.py -j generate_input_prompts_vB/generate_input_prompts_v3.json
python generate_input_prompts.py -j generate_input_prompts_vB/generate_input_prompts_v4.json
python generate_input_prompts.py -j generate_input_prompts_vB/generate_input_prompts_v5.json
```

vB:  Prompt selection
```bash

# generation
python generate_lm_output.py -j ./generate_lm_output_vB/generate_lm_output_gpt-j-6b.json
python generate_lm_output.py -j ./generate_lm_output_vB/generate_lm_output_bloom-7b1.json
python generate_lm_output.py -j ./generate_lm_output_vB/generate_lm_output_opt-6.7b.json
python generate_lm_output.py -j ./generate_lm_output_vB/generate_lm_output_opt-13b.json
python generate_lm_output.py -j ./generate_lm_output_vB/generate_lm_output_opt-30b.json

# filter
python discard_on_selfBLEU.py -j ./discard_on_selfBLEU_vB_v2/discard_on_selfBLEU4.json
python discard_on_criteria.py -j ./discard_on_criteria/discard_on_criteria_vB.json

# perplexity calculation
python calculate_perplexity.py -j ./calculate_perplexity_vB_v2/calculate_perplexity_gptj6b.json
python calculate_perplexity.py -j ./calculate_perplexity_vB_v2/calculate_perplexity_bloom-7b1.json
python calculate_perplexity.py -j ./calculate_perplexity_vB_v2/calculate_perplexity_opt-6.7b.json
python calculate_perplexity.py -j ./calculate_perplexity_vB_v2/calculate_perplexity_opt-13b.json
python calculate_perplexity.py -j ./calculate_perplexity_vB_v2/calculate_perplexity_opt-30b.json

# join perpleixty results
python join_results_perplexity.py -j ./join_results_perplexity_vB/join_results_perplexity.json

# join sentences to be analyzed with LIWC
python join_all_sentences_v2.py -j ./join_all_sentences_vB_v2/join_all_sentences_v2.json

# Process LIWC results 
python process_LIWC_results.py -j ./process_LIWC_results_vB/process_LIWC_results_v1_vB.json
python process_LIWC_results.py -j ./process_LIWC_results_vB/process_LIWC_results_v2_vB.json

```

vC: Decoding strategy selection
```bash
# generation
python get_sentences_from_output_vC.py ./get_sentences_from_output_lm_vC/get_sentences_from_output_lm_gpt-j-6b.json
python get_sentences_from_output_vC.py ./get_sentences_from_output_lm_vC/get_sentences_from_output_lm_bloom-7b1.json
python get_sentences_from_output_vC.py ./get_sentences_from_output_lm_vC/get_sentences_from_output_lm_opt-6.7b.json
python get_sentences_from_output_vC.py ./get_sentences_from_output_lm_vC/get_sentences_from_output_lm_opt-13b.json
python get_sentences_from_output_vC.py ./get_sentences_from_output_lm_vC/get_sentences_from_output_lm_opt-30b.json

python get_statistics_messages_generated.py -j ./get_statistics_messages_generated_vC/get_statistics_messages_generated.json

# BLEU-4
python postprocess_on_selfBLEU.py -j ./postprocess_on_selfBLEU4_vC/postprocess_on_selfBLEU4.json

# filter
python discard_on_selfBLEU.py -j ./discard_on_selfBLEU_vC/discard_on_selfBLEU4.json
python discard_on_criteria.py -j ./discard_on_criteria/discard_on_criteria_vC.json

# perplexity
python calculate_perplexity.py -j ./calculate_perplexity_vC/calculate_perplexity_gptj6b.json
python calculate_perplexity.py -j ./calculate_perplexity_vC/calculate_perplexity_bloom-7b1.json
python calculate_perplexity.py -j ./calculate_perplexity_vC/calculate_perplexity_opt-6.7b.json
python calculate_perplexity.py -j ./calculate_perplexity_vC/calculate_perplexity_opt-13b.json
python calculate_perplexity.py -j ./calculate_perplexity_vC/calculate_perplexity_opt-30b.json

# join perpleixty results
python join_results_perplexity.py -j ./join_results_perplexity_vC/join_results_perplexity_vC.json

# join sentences to be analyzed with LIWC
python join_all_sentences_v2_vC.py -j ./join_all_sentences_vC/join_all_sentences_vC.json

# analysis of LIWC results
python process_LIWC_results.py -j ./process_LIWC_results_vC/process_LIWC_results_v1.json
python process_LIWC_results.py -j ./process_LIWC_results_vC/process_LIWC_results_v2.json

```

Process ChatGPT messages:
```bash
python get_sentences_from_output_ChatGPT.py -j get_sentences_from_output_ChatGPT/get_sentences_from_output_prompt_v4_ChatGPT.json
python postprocess_on_selfBLEU.py -j ./postprocess_on_selfBLEU_ChatGPT/postprocess_on_selfBLEU4.json
python discard_on_selfBLEU.py -j ./discard_on_selfBLEU_ChatGPT/discard_on_selfBLEU4.json
python discard_on_criteria.py -j discard_on_criteria_ChatGPT.json

# join sentences to be analyzed with LIWC
python join_all_sentences_v2_vC.py -j ./join_all_sentences/join_all_sentences_ChatGPT/join_all_sentences_ChatGPT.json
```



