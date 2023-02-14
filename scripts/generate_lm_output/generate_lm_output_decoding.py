""" generate_lm_output_decoding.py
    It loads the Language models and generates messages according to specified parameters. It allows to specify different decoding versions.
        
    python generate_lm_output_decoding.py -j ./generate_lm_output_vC/generate_lm_output_decoding_gpt-j-6b.json
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

def get_messages(config_json):
    """It get a pandas dataframe from a csv file

    Args:
        data_file_path (path_type: csv): csv file wit the messages
    
    Returns:
        pandas dataframe: the mean
    """
    
    data_file_path = config_json["file_path"] 
    
    df_data = pd.read_csv(data_file_path, index_col=0) 
    df_data
    # print("Running")
    return df_data    


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
    
def generate_messages(config_json):

    model = load_model(config_json)
    tokenizer = get_tokenizer(config_json)

    for csv_filepath in config_json["list_filenames"]:
        print("Processing", csv_filepath)

        for enum_decoding, decoding_config in enumerate(config_json["decoding"]):
            l_messages = []

            df_prompts = pd.read_csv(csv_filepath, index_col=0)
            col_name = df_prompts.columns[0]
            
            # version_decoding = enum_decoding
            version_decoding = enum_decoding + 2

            print(version_decoding, decoding_config)
            
            for i in range(df_prompts.shape[0]):
            # for i in range(5):

                print(f"{i+1}/{df_prompts.shape[0]}")
                input_ids = tokenizer.encode(str(df_prompts.loc[i,col_name]), return_tensors='pt').cuda()

                output = model.generate(input_ids, do_sample=True,
                                        max_length=decoding_config["max_length"], top_p=decoding_config["top_p"],
                                        top_k=decoding_config["top_k"], temperature=decoding_config["temperature"] )
        
                output_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                
                l_messages.append(output_decoded)
                
            series_prompts = pd.Series(l_messages)

            get_series_to_csv(series_prompts, csv_filepath, version_decoding, "output", config_json)
           
def get_series_to_csv(series_data, filepath, enum_decoding, column_name, config_json):

    path_output = config_json["path_output"]
    
    os.makedirs(path_output, exist_ok = True)
    # get path
    
    series_data = series_data.rename(f"{column_name}")
    df_temp = pd.DataFrame(series_data)
    
    path_output = config_json["path_output"]
    
    if config_json["language_model"] == "EleutherAI/gpt-j-6B":
        suffix_model = "gpt-j-6B"
    elif config_json["language_model"] == "bigscience/bloom-7b1":
        suffix_model = "bloom-7b1"
    elif config_json["language_model"] == "facebook/opt-6.7b":
        suffix_model = "opt-6.7b"
    elif config_json["language_model"] == "facebook/opt-13b":
        suffix_model = "opt-13b"
    elif config_json["language_model"] == "facebook/opt-30b":
        suffix_model = "opt-30b"

    # file_name from filepath
    filename = os.path.basename(filepath)
    
    file_name = f"{filename[:-4]}_gen_{suffix_model}_decoding_v{enum_decoding+1}.csv"   
    file_path = os.path.join(path_output, file_name)

    df_temp.to_csv(file_path, encoding = "utf-8")

    return None

def main():
    args = get_parser()
    config_json = get_json(args)

    series_messages = generate_messages(config_json)

    print("Hello World")

if __name__ == "__main__":
    main()
