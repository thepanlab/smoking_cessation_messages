""" prepare_dataset_to_prompts.py
    This script takes as input the csv file and split it in two or three parts.
    Then, it stores them as csv. Additionally, it stores them with prompt format. 

    python prepare_dataset_to_prompts -j prepare_dataset_to_prompts_v1.json
"""
import argparse
import json
from multiprocessing.sharedctypes import Value
import pandas as pd
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

def get_messages(config_json):
    """It get a pandas dataframe from a csv file

    Args:
        data_file_path (path_type: csv): csv file wit the messages
    
    Returns:
        pandas dataframe: the mean
    """
    
    data_file_path = config_json["file_path"] 
    
    pd_data = pd.read_csv(data_file_path) 
    pd_data
    print("Running")
    return pd_data    

def split_data(pd_data, config_json):
    number_messages = config_json["number_of_messages"]
    seed = config_json["seed"]
 
    bool_split = False
    if config_json["split"] == "True":
        number_splits= config_json["number_of_splits"]
        bool_split = True 
        if number_splits not in  [2,3]:
            raise ValueError(f"number of splits allowed are 2 or 3 only. The value was received {number_splits}")
            
    elif config_json["split"] == "False":
        pass
    
    # Get messages and randomize
    series_message = pd_data.loc[0:number_messages-1,"Message"]
    series_message_random = series_message.sample(frac=1, random_state= seed).reset_index(drop = True)
    
    l_series = []
    
    # getting division cuts
    if bool_split == True:
        if number_splits == 2:
            cut_point = [round(0.80*number_messages)]
            series_train = series_message_random[:cut_point[0]]
            series_train = series_train.reset_index(drop=True)

            series_validation = series_message_random[cut_point[0]:number_messages]
            series_validation = series_validation.reset_index(drop=True)
            
            l_series.extend([series_train, series_validation])
        
        elif number_splits == 3:
            cut_point = [round(0.60*number_messages),round(0.80*number_messages)]
            series_train = series_message_random[:cut_point[0]]
            series_train = series_train.reset_index(drop=True)
            
            series_validation = series_message_random[cut_point[0]:cut_point[1]]
            series_validation = series_validation.reset_index(drop=True)
            
            series_test = series_message_random[cut_point[1]:number_messages]
            series_test = series_test.reset_index(drop=True)
            
            l_series.extend([series_train, series_validation, series_test])

    
    elif bool_split == False:
        l_series.append(series_message_random)
    
    return l_series

def get_series_parts_to_csv(list_of_series_of_text, column_name, config_json):
    
    path_output = config_json["path_output_split"]
    os.makedirs(path_output, exist_ok = True)
    # get path
    
    for i, serie_text in enumerate(list_of_series_of_text):
        serie_text = serie_text.rename(f"{column_name}")
        df_temp = pd.DataFrame(serie_text)
        
        if i == 0:
            file_name = "train.csv"
        elif i == 1:
            file_name = "validation.csv"
        elif i == 2:
            file_name = "test.csv"
        
        file_path = os.path.join(path_output, file_name)

        df_temp.to_csv(file_path,encoding = "utf-8")

def get_series_prompt_format_to_csv(list_of_series_of_text, column_name, config_json):
    
    prefix_name = config_json["prefix_name"]
    path_output = config_json["path_output_prompt_format"]
    os.makedirs(path_output, exist_ok = True)
    
    for i, serie_text in enumerate(list_of_series_of_text):
        serie_text = serie_text.rename(f"{column_name}")
        d_temp = pd.DataFrame(serie_text)
        
        if i == 0:
            file_name = f"{prefix_name}_train.csv"
        elif i == 1:
            file_name = f"{prefix_name}_validation.csv"
        elif i == 2:
            file_name = f"{prefix_name}_test.csv"
        
        file_path = os.path.join(path_output, file_name)

        d_temp.to_csv(file_path,encoding = "utf-8")

def get_message_prompt_format(l_series, config_json):
    
    location_valid_options = ["start","end"]

    if config_json["location_prompt"] not in location_valid_options:
        raise ValueError(f"location_prompt: {config_json['location_prompt']}. Values accepted: {location_valid_options}")
    
    config_json["prompt message"]
    config_json["symbol"]
    config_json["number_of_messages_per_prompt"]
    l_parts = []

    for serie in l_series:
               
        l_prompts = []
        index_start_mssg = 0

        config_json["number_of_messages_per_prompt"]
        n_messages_serie = serie.shape[0]       
              
        while index_start_mssg + config_json["number_of_messages_per_prompt"] - 1 < n_messages_serie:
            
            # if index_start_mssg + config_json["number_of_messages_per_prompt"] - 1 >= n_messages_serie:
            #     break
            
            l_prompt = []
            if config_json["location_prompt"] == "start":
                l_prompt.append(config_json["prompt message"])
            
            # loop of 5 or less messages add to l_prompt
            index_prompt = 0
                      
            while  index_start_mssg + index_prompt < n_messages_serie and index_prompt < 5:
                if config_json["symbol"] == "numbers":
                    symbol = f"{index_prompt+1}."
                else:
                    symbol = config_json["symbol"]

                l_prompt.append(f"{symbol} {serie[index_start_mssg + index_prompt]}")
                index_prompt += 1

            if config_json["location_prompt"] == "end":
                l_prompt.append(config_json["prompt message"])

            l_prompts.append("\n\n".join(l_prompt))

            # create series

            index_start_mssg += config_json["number_of_messages_per_prompt"]

        l_parts.append(pd.Series(l_prompts))
    
    return l_parts

def remove_messages(pd_messages, config_json):
    # read row by row, discard messages with given word
    l_words = config_json["list_to_filter"]

    pd_messages_filtered = pd_messages.copy() # deep by defaut
    l_indices_to_remove = []
    for i in range(config_json["number_of_messages"]):

        pd_messages.loc[i, "Message"]

    # for loop
    # lower case
    # split by spaces
    # if messages in row has the word removed




def main():
    args = get_parser()
    config_json = get_json(args)
    pd_messages = get_messages(config_json)
    l_parts = split_data(pd_messages, config_json)

    # pd_messages_filtered = remove_messages(pd_messages, config_json)

    get_series_parts_to_csv(l_parts, "Message", config_json)
    
    # l_prompt_parts = get_message_prompt_format(l_parts, config_json)
    get_series_prompt_format_to_csv(l_prompt_parts, "prompts", config_json)
    
    print("Hello World")
    
if __name__ == "__main__":
    main()
