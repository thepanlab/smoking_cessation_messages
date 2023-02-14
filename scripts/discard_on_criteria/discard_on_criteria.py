""" discard_on_criteria.py

    It allows to discard messages according to some patters

    python discard_on_criteria.py -j ./discard_on_criteria/discard_on_criteria_vB.json
"""

import argparse
import copy
import json
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import multiprocessing
from joblib import Parallel, delayed
import os

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

def get_n_parallel_processes(config_json):
    """
    Obtains number of cores to be used
    Arguments:
        config_json (dict): 
    Returns:
        int: number of cores that will be used 
    Raise:
        ValueError: if number of cores provided is larger than maximum
                    if number of cores is less than zero
    """
    num_cores = multiprocessing.cpu_count()

    n_parallel_processes = config_json["n_parallel_processes"]

    if n_parallel_processes == "all":
        return num_cores
    elif n_parallel_processes > num_cores:
        raise ValueError(f"Number of parallel process greater than the actual number of cores :{num_cores}")
    elif n_parallel_processes < 0:
        raise ValueError("Number of parallel process lower than zero")
    else:
        return config_json.n_parallel_processes

def get_boolean_criteria(message, list_criteria):
    # split the messages
    l_split = message.split()
    
    b_presence = False

    for criterion in list_criteria:
        if criterion == "_":          
            for word_split in l_split:
                b_presence = b_presence or (criterion[0] in word_split)
        elif criterion == "less_than_six_words":
            b_presence = ( len(l_split) < 6 )
        else:
            b_presence = b_presence or (criterion in l_split)

    return b_presence
        
def criteria_on_dataframe(df_messages, name, list_criteria):

    df_messages[name] = df_messages.apply(lambda x: get_boolean_criteria(x["Message"], list_criteria), axis=1)   

    return df_messages

def apply_criteria(config_json):
    
    criteria_names = config_json["criteria_names"]

    os.makedirs(config_json["path_output"], exist_ok = True)

    # statistics
    
    l_columns = ["filename", "n_total"]
        
    l_columns.extend([f"n_{criterion}" for criterion in criteria_names])

    l_columns.extend(["n_remaining","percentage"])

    df_statistic = pd.DataFrame(columns = l_columns)
    
    for filepath in config_json["list_filenames"]:
        df_messages = pd.read_csv(filepath, index_col = 0)
        for name, list_criteria in zip(config_json["criteria_names"], config_json["criteria"]):
            df_messages = criteria_on_dataframe(df_messages, name, list_criteria)
        
        filename = os.path.basename(filepath)
        filepath_output = os.path.join(config_json["path_output"],f"{filename[:-4]}_criteria_value.csv")

        df_messages.to_csv(filepath_output,encoding = "utf-8")
        # discard version
        str_query = ""
        for i, criterion_name in enumerate(criteria_names):
            str_query = str_query + f"{criterion_name} == False"
            if i < len(criteria_names) - 1 :
                str_query = str_query + " and "

        df_messages_discarded = df_messages.query(f"{str_query}")
        filepath_output = os.path.join(config_json["path_output"],f"{filename[:-4]}_criteria_remaining.csv")
        df_messages_discarded.to_csv(filepath_output,encoding = "utf-8")
        
        # create dictionary
        dict_df_row = {"filename":[filename]}

        n_generated_messages = df_messages.shape[0]
        dict_df_row["n_total"] = [n_generated_messages]

        for i, criterion_name in enumerate(criteria_names):
            str_query = f"{criterion_name} == True"
            dict_df_row[f"n_{criterion_name}"] = [ df_messages.query(str_query).shape[0] ]

        n_reduced_messages = df_messages_discarded.shape[0]
        dict_df_row["n_remaining"] = [n_reduced_messages]
        dict_df_row["percentage"] = [n_reduced_messages/n_generated_messages]

        df_temp = pd.DataFrame(dict_df_row)

        df_statistic = pd.concat([df_statistic, df_temp], ignore_index=True)

    filepath_output = os.path.join(config_json["path_output"], "statistics_discard.csv")
    df_statistic.to_csv(filepath_output,encoding = "utf-8")

def main():

    args = get_parser()
    config_json = get_json(args)

    # filepath = config_json["list_filenames"][0]

    # df_message_one = pd.read_csv(filepath, index_col = 0)  

    # list_criteria = ["app","apps","application"]
    # message = "The app is the solution to your problems"
    # get_boolean_criteria(message, list_criteria, config_json)
    # message = "Stand up"
    # list_criteria = ["less-than-six-words"]
    # get_boolean_criteria(message, list_criteria)

    apply_criteria(config_json)

if __name__ == "__main__":
    main()