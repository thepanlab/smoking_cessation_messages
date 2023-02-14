""" get_statistics_messages_generated.py

    It produces two files: 
        * `statistics_summary.csv`: average number of messages per model/version
        * `statistics_index_prompt_level.csv`: number of messages per prompt

    python get_statistics_messages_generated.py -j ./get_statistics_messages_generated_vB_v2/get_statistics_messages_generated_v2.json
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
import re

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

def get_number_messages_prompt_level(filepath, config_json):

    df_messages = pd.read_csv(filepath, index_col = 0)

    df_statistics_prompt_level = pd.DataFrame(columns = ["filename","index_prompt",
                                                         "number_messages"])
    filename = os.path.basename(filepath)

    a_index_prompt = df_messages["index_prompt"].unique()

    for index in a_index_prompt:
        df_subset = df_messages.query(f"index_prompt == {index}")
        n_elements = df_subset.shape[0]

        df_temp = pd.DataFrame({"filename":[filename], 
                        "index_prompt":[index],
                        "number_messages":[n_elements]})

        df_statistics_prompt_level = pd.concat([df_statistics_prompt_level, df_temp], ignore_index=True)

    return df_statistics_prompt_level

def get_number_messages_summary(filepath, config_json):
    l_number_messages = []

    df_messages = pd.read_csv(filepath, index_col = 0)
    for i in df_messages["index_prompt"].unique():
        df_index_prompt = df_messages.query(f"index_prompt == {i}")
        l_number_messages.append(df_index_prompt.shape[0])

    return df_messages.shape[0], np.mean(l_number_messages), np.std(l_number_messages, ddof = 1)/np.sqrt(len(l_number_messages))
   
def get_df_statistics(config_json):
    os.makedirs(config_json["path_output"], exist_ok = True)

    df_statistics_prompt_level = pd.DataFrame(columns = ["filename","index_prompt",
                                                         "number_messages"])
    df_statistics_summary = pd.DataFrame(columns = ["filename","total_number_messages",
                                                    "number_messages_mean","number_messages_std_err"])

    for filepath in config_json["list_filenames"]:

        filename = os.path.basename(filepath)

        n_messages, n_messages_mean, n_messages_std_err = get_number_messages_summary(filepath, config_json)
        df_prompt_temp = get_number_messages_prompt_level(filepath, config_json)
        
        df_temp = pd.DataFrame({"filename":[filename],
                                "total_number_messages":[n_messages],
                                "number_messages_mean":[n_messages_mean],
                                "number_messages_std_err":[n_messages_std_err]})

        df_statistics_prompt_level = pd.concat([df_statistics_prompt_level, df_prompt_temp], ignore_index=True)
        df_statistics_summary = pd.concat([df_statistics_summary, df_temp], ignore_index=True)

    path_output_file_prompt_level = os.path.join(config_json["path_output"],
                                        f"statistics_index_prompt_level.csv")
    path_output_file_summary = os.path.join(config_json["path_output"],
                                        f"statistics_summary.csv")

    df_statistics_prompt_level.to_csv(path_output_file_prompt_level)
    df_statistics_summary.to_csv(path_output_file_summary)

def main():

    args = get_parser()
    config_json = get_json(args)

    get_df_statistics(config_json)

if __name__ == "__main__":
    main()
