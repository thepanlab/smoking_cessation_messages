""" get_sentences_from_output_lm.py

    It allows to process sentences with vA of prompts

    python get_sentences_from_output_lm.py -j get_sentences_from_output_lm_gpt-j-6B.json
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

def get_file_to_sentences(config_json):
    
    path_output = config_json["path_output"]
    
    os.makedirs(path_output, exist_ok = True)
    
    for csv_filepath in config_json["list_filenames"]:
        df_generated = pd.read_csv(csv_filepath, index_col=0)
        
        filename = os.path.basename(csv_filepath)
        
        l_messages = []
        l_index = []

        if "v1" in filename or "v3" in filename:
            # split each row by "*"
            for i_row in range(df_generated.shape[0]):
                messages = df_generated.iloc[i_row,0]        
                l_split_messages = messages.split("* ")
                # The first 5 messages are the provided examples
                # The last message is discarded
                l_messages_produced_row = [ split_message[:-2] for split_message in l_split_messages[6:-2] ]
                l_messages.extend(l_messages_produced_row)                  

                l_index.extend([i_row for i in range(len(l_messages_produced_row))])
                
        elif "v2" in filename or "v4" in filename:

            l_messages = []
            l_index = []

            for i_row in range(df_generated.shape[0]):
                l_start_search = []
                l_end_search = []
                index_numeric = 6
                while True:
                    x = re.search(rf"\n\n{index_numeric}.", df_generated.iloc[i_row,0])

                    if x is None:
                        break

                    index_numeric+=1

                    l_start_search.append(x.start())
                    l_end_search.append(x.end())
                
                for i_regex in range(len(l_start_search) - 1):
                    l_messages.append(df_generated.iloc[i_row,0][l_end_search[i_regex]+1:l_start_search[i_regex+1]])
                    l_index.append(i_row)

                print("Hello World")

        elif "v5" in filename:

            l_messages = []
            l_index = []

            for i_row in range(df_generated.shape[0]):
                x = re.search("Write similar messages to the previous ones:", df_generated.iloc[i_row,0])
                
                l_split_messages = df_generated.iloc[i_row,0][x.end():].split("* ")

                # first element and last sentence (most certain not finished) discarded
                l_messages_produced_row = [ message_temp[:-2] for message_temp in l_split_messages[2:-2] if len(message_temp[:-2])> 0]

                l_messages.extend(l_messages_produced_row)
                l_index.extend([i_row for i in range(len(l_messages_produced_row))])
                
            print("Hello World")

        elif "v6" in filename:
            l_messages = []
            l_index = []

            for i_row in range(df_generated.shape[0]):
                x1 = re.search("Write similar messages to the previous ones:", df_generated.iloc[i_row,0])
            
                output_subset = df_generated.iloc[i_row,0][x1.end():]

                l_start_search = []
                l_end_search = []
                index_numeric = 1
                while True:
                    x2 = re.search(rf"\n\n{index_numeric}.", output_subset)

                    if x2 is None:
                        break

                    index_numeric+=1

                    l_start_search.append(x2.start())
                    l_end_search.append(x2.end())
                
                for i_regex in range(len(l_start_search) - 1):
                    l_messages.append(output_subset[l_end_search[i_regex]+1:l_start_search[i_regex+1]])
                    l_index.append(i_row)

                print("Hello World")
            
        # Store as pandas dataframe
        filename = os.path.basename(csv_filepath)

        df_temp = pd.DataFrame( {"message":l_messages, "index_prompt":l_index})

        file_name = f"{filename[:-4]}_to_sentences.csv"   
        file_path = os.path.join(path_output, file_name)
        
        df_temp.to_csv(file_path, encoding = "utf-8")
        
    return None

def main():
    args = get_parser()
    config_json = get_json(args)

    get_file_to_sentences(config_json)

if __name__ == "__main__":
    main()
