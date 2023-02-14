""" generate_input_prompts.py
    This script takes as input the train or validation messages and generates
    prompts with specific number of example and format taken randomly from the input file
    
    python generate_input_prompts.py -j ./generate_input_prompts_vB/generate_input_prompts_v1.json
"""

import argparse
import json
from multiprocessing.sharedctypes import Value
import pandas as pd
import os
import numpy as np 

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

def get_prompts(config_json):

    df_data = get_messages(config_json)

    col_name = df_data.columns[0]

    a_indices = np.arange(0, df_data.shape[0], 1)
        
    symbol = config_json["symbol"]
    separation_lines = config_json["separation_lines"]

    l_prompts = []

    for i in range(config_json["number_of_prompts"]):
        print(i, end="\r")
        a_indices_shuffled = np.copy(a_indices)
        np.random.seed(config_json["seed"]*i)
        np.random.shuffle(a_indices_shuffled)
        
        l_prompt = []

        if config_json["location_prompt"] == "start":
            l_prompt.append(config_json["prompt message"])

        for j in range(config_json["number_of_messages_per_prompt"]):
            if config_json["symbol"] == "numbers":
                symbol = f"{j+1}."
            elif config_json["symbol"] == "message numbers":
                symbol = f"Message {j+1}:"
            else:
                symbol = config_json["symbol"]
            l_prompt.append(f"{symbol} {df_data.loc[a_indices_shuffled[j], col_name]}")

        if config_json["location_prompt"] == "end":
            l_prompt.append(config_json["prompt message"])

        if config_json["symbol"] == "numbers" and config_json["location_prompt"] == "start":
            symbol = f"{config_json['number_of_messages_per_prompt'] + 1}."
        elif config_json["symbol"] == "numbers" and config_json["location_prompt"] == "end":
            symbol = "1."
        elif config_json["symbol"] == "message numbers" and config_json["location_prompt"] == "start":
            symbol = f"Message {config_json['number_of_messages_per_prompt'] + 1}:"            
        else:
            symbol = config_json["symbol"]

        l_prompt.append(f"{symbol}")

        l_prompts.append(separation_lines.join(l_prompt))
    series_prompts = pd.Series(l_prompts)

    return series_prompts

def get_series_to_csv(series_data, column_name, config_json):

    path_output = config_json["path_output"]
    filename_output = config_json["filename_output"]
    
    os.makedirs(path_output, exist_ok = True)
    # get path
    
    series_data = series_data.rename(f"{column_name}")
    d_temp = pd.DataFrame(series_data)
    
    file_name = f"{filename_output}.csv"   
    file_path = os.path.join(path_output, file_name)

    d_temp.to_csv(file_path,encoding = "utf-8")

    return None

def main():
    args = get_parser()
    config_json = get_json(args)
    pd_messages = get_messages(config_json)
    s_prompts = get_prompts(config_json)
    get_series_to_csv(s_prompts, "prompts", config_json)

    print("Hello World")

if __name__ == "__main__":
    main()
