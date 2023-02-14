''' join_all_sentences_v2.py
    This scripts allows to join the original dataset with the sentences produced
    by the Language Models: GPT-J-6B, Bloom 7B1, and OPT 6.7b for decoding selection
    
    It allows to add category: original, train, val

    python join_all_sentences_v2_vC.py -j ./join_all_sentences_vC/join_all_sentences_v2_vC.json
'''

import argparse
import json
from multiprocessing.sharedctypes import Value
import pandas as pd
import os
import numpy as np
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

def join_files(config_json):
    
    # Read original file
    df_dataset = pd.read_csv(config_json["original_data"])
    # Select subset
    df_dataset_subset_columns = df_dataset.loc[:df_dataset.shape[0]-2,["Message","ID"]]
    df_dataset_subset_columns["model"] = "original"
    df_dataset_subset_columns["type"] = "original"
    print("Hello World")

    df_all = pd.DataFrame()
    df_all = pd.concat([df_all, df_dataset_subset_columns], ignore_index=True)
    
    df_all_2 = pd.DataFrame()

    for path_csv in config_json["partition_data"]:
        df_temp = pd.read_csv(path_csv, index_col=0)
        
        filename = os.path.basename(path_csv)
        df_temp = df_temp.rename(columns={"messages": "Message"})
        if "train" in filename:
            df_temp["model"] = "original-train"
        elif "validation" in filename:
            df_temp["model"] = "original-validation"
        df_all_2 = pd.concat([df_all_2, df_temp], ignore_index=True)
        
    for path_csv in config_json["directory_paths"]:
        df_temp = pd.read_csv(path_csv, index_col=0)
        # Rename message to Message
        df_temp = df_temp.rename(columns={"message": "Message"})
        filename = os.path.basename(path_csv)

        if "gpt-j-6B" in filename:
            model_name = "gpt-j-6B"
        elif "bloom-7b1" in filename:
            model_name = "bloom-7b1"
        elif "opt-6.7b" in filename:
            model_name = "opt-6.7b"
        elif "opt-13b" in filename:
            model_name = "opt-13b"
        elif "opt-30b" in filename:
            model_name = "opt-30b"
        else:
            raise ValueError("Model not supported. Models supported gpt-j-6b, bloom-7b1, opt-6.7b, opt-13b, and opt-30b.")
        
        m = re.search('(?<=v)[0-9]+', filename)

        if m is None:
            raise ValueError("filename doesn't contain a version")
        
        # Extract decoding version
        m2 = re.search('(?<=decoding_v)[0-9]+', filename)


        # add v1 and model name
        model_prompt_version = f"{model_name}_prompt_v{m.group(0)}_decoding_v{m2.group(0)}"
        df_temp["model"] = model_name
        df_temp["prompt"] = f"v{m.group(0)}"
        df_temp["decoding"] = f"v{m2.group(0)}"
        df_temp["type"] = model_prompt_version

        df_all = pd.concat([df_all, df_temp], ignore_index=True)
        df_all_2 = pd.concat([df_all_2, df_temp], ignore_index=True)

    os.makedirs(config_json["path_directory_output"], exist_ok = True)

    path_output_file = os.path.join(config_json["path_directory_output"], "joined_results.csv")
    path_output_file_v2 = os.path.join(config_json["path_directory_output"], "joined_results_v2.csv")

    df_all.to_csv(path_output_file, encoding = "utf-8")
    df_all_2.to_csv(path_output_file_v2, encoding = "utf-8")

    print("Hello World")

def main():
    args = get_parser()
    config_json = get_json(args)

    join_files(config_json)

if __name__ == "__main__":
    main()