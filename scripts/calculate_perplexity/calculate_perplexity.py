""" calculate_perplexity.py

    It allows to calculate the perplexity according to a specific language model.

    python calculate_perplexity.py -j ./calculate_perplexity_vB_v2/calculate_perplexity_gptj6b.json
"""

import argparse
import json
import pandas as pd
import os
import numpy as np
import time
import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import transformers
from tqdm import tqdm
from termcolor import colored

def get_parser():
    """
    Obtains arguments parser

    Arguments:
        None
    Returns:
        ArgumentParset args
    """
    # https://www.loekvandenouweland.com/content/using-json-config-files-in-python.html
    parser = argparse.ArgumentParser()

    parser.add_argument("-j","--load_json", required = True,
            help='Load settings from file in json format.')

    args = parser.parse_args()

    return args
# print(parser)

def get_json(args):
    """
    Obtains configurations stores in json file

    Arguments:
        ArgumentParser: args
    Returns:
        dict config_json: it contains all the parameters from json file
    """
    with open(args.load_json) as config_file:
        config_json = json.load(config_file)

    return config_json

def load_model(config_json):
    print("Loading model - Start")
    if config_json["language_model"] == "EleutherAI/gpt-j-6B":
        if torch.cuda.is_available():
            model_loaded =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).cuda()
        else:
            model_loaded =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
            
    elif config_json["language_model"] == "bigscience/bloom-7b1":
        model_loaded = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1",
                                                    device_map="auto",
                                                    torch_dtype="auto")
        
    elif config_json["language_model"] == "facebook/opt-6.7b":
        model_loaded = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16).cuda()

    elif config_json["language_model"] == "facebook/opt-13b":
        model_loaded = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16).cuda()
    elif config_json["language_model"] == "facebook/opt-13b":
        model_loaded = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16).cuda()
    elif config_json["language_model"] == "facebook/opt-30b":
        # if torch_dtype is not used as parameter, the default value is torch.float16
        # If ran like this, it places more layers on GPU. idk why
        # model_loaded = AutoModelForCausalLM.from_pretrained("facebook/opt-30b", device_map="auto",
        #                                                      torch_dtype=torch.float16,
        #                                                      offload_folder='./offload_folder')
        model_loaded = AutoModelForCausalLM.from_pretrained("facebook/opt-30b", device_map="auto",
                                                            offload_folder='./offload_folder')
        print(colored("model.dtype: {model.dtype}","green"))
        print(model_loaded.hf_device_map)
        print(colored("For model opt-30b, option device_map='auto' automatically determine where to put each layer to maximize the use of your fastest devices (GPUs) and offload the rest on the CPU", "green"))
        print(colored("model.hf_device_map:","yellow"))
        print(model_loaded.hf_device_map)
    print("Loading model - End")
       
    return model_loaded

def get_tokenizer(config_json):
    if config_json["language_model"] == "EleutherAI/gpt-j-6B":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    elif config_json["language_model"] == "bigscience/bloom-7b1":
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    elif config_json["language_model"] == "facebook/opt-6.7b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
    elif config_json["language_model"] == "facebook/opt-13b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b", use_fast=False)
    elif config_json["language_model"] == "facebook/opt-30b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
    return tokenizer

def get_max_length(model,config_json):
    if config_json["language_model"] == "EleutherAI/gpt-j-6B":
        max_length = model.config.n_positions # 2048
    elif config_json["language_model"] == "bigscience/bloom-7b1":
        # Sequence length of 2048 tokens used https://huggingface.co/bigscience/bloom
        # let's use have
        max_length = 2048
    elif config_json["language_model"] == "facebook/opt-6.7b":
        max_length = model.config.max_position_embeddings # 2048
    elif config_json["language_model"] == "facebook/opt-13b":
        max_length = model.config.max_position_embeddings # 2048
    elif config_json["language_model"] == "facebook/opt-30b":
        max_length = model.config.max_position_embeddings # 2048
    return max_length

def get_perplexity_one_file(filepath, model, tokenizer, config_json):
    # Based on the code at https://huggingface.co/docs/transformers/perplexity

    max_length = get_max_length(model, config_json)

    device = "cuda"
    
    df_messages = pd.read_csv(filepath, index_col=0)
    # print(df_val.shape)

    column_to_analyze = config_json["column_to_analyze"]

    list_rows = df_messages[column_to_analyze].to_list()
    str_rows_joined = "\n\n".join(list_rows)

    encodings = tokenizer(str_rows_joined, return_tensors="pt")

    stride = config_json["stride"]
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl.detach().cpu().numpy()


def get_perplexity_files(config_json):
# path_messages, model, tokenizer_name, column_name

    model = load_model(config_json)
    tokenizer = get_tokenizer(config_json)  

    df = pd.DataFrame(columns = ['filename', 'perplexity'])

    for csv_filepath in config_json["list_filenames"]:
        ppl = get_perplexity_one_file(csv_filepath, model, tokenizer, config_json)

        filename = os.path.basename(csv_filepath)

        df = df.append({'filename' : filename, 'perplexity' : ppl}, 
                        ignore_index = True)

    os.makedirs(config_json["path_output"], exist_ok = True)
   
    if config_json["language_model"] == "EleutherAI/gpt-j-6B":
        filepath_output = os.path.join(config_json["path_output"],"perplexity_results_gptj6b.csv")
        df.to_csv(filepath_output,encoding = "utf-8")
    elif config_json["language_model"] == "bigscience/bloom-7b1":
        filepath_output = os.path.join(config_json["path_output"],"perplexity_results_bloom-7b1.csv")
        df.to_csv(filepath_output,encoding = "utf-8")
    elif config_json["language_model"] == "facebook/opt-6.7b":
        filepath_output = os.path.join(config_json["path_output"],"perplexity_results_opt-6.7b.csv")
        df.to_csv(filepath_output,encoding = "utf-8")
    elif config_json["language_model"] == "facebook/opt-13b":
        filepath_output = os.path.join(config_json["path_output"],"perplexity_results_opt-13b.csv")
        df.to_csv(filepath_output,encoding = "utf-8")
    elif config_json["language_model"] == "facebook/opt-30b":
        filepath_output = os.path.join(config_json["path_output"],"perplexity_results_opt-30b.csv")
        df.to_csv(filepath_output,encoding = "utf-8")
def main():

    args = get_parser()
    config_json = get_json(args)
    
    get_perplexity_files(config_json)
    # path_model = "EleutherAI/gpt-j-6B"
    # model = get_model(path_model)
    # tokenizer_name = "EleutherAI/gpt-j-6B"
    # path_messages = './sentences/gptj6b_4_processed/sentences_processed.csv'
    # ppl = get_perplexity(path_messages, model, tokenizer_name, "Messages")
    # print(ppl)
    
if __name__ == "__main__":
    main()