""" get_sentences_from_output_lm_vB_v2.py

    It allows to process sentences with vB of prompts. Based on feedback improvements to the process. 

    Improvements:
    * 

    python get_sentences_from_output_lm_vB_v2.py -j ./get_sentences_from_output_lm_vB_v2/get_sentences_from_output_lm_gpt-j-6B.json
"""

import argparse
import json
import pandas as pd
import os
import numpy as np
import time
import torch
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
    number_of_messages_per_prompt = config_json["number_of_messages_per_prompt"]
    os.makedirs(path_output, exist_ok = True)
    
    for csv_filepath in config_json["list_filenames"]:
        df_generated = pd.read_csv(csv_filepath, index_col=0)
        
        filename = os.path.basename(csv_filepath)

        print(filename)
        
        l_messages = []
        l_index = []
        l_message_position = []

        if "v1" in filename or "v2" in filename or "v3" in filename:

            for i_row in range(df_generated.shape[0]):
                l_start_search = []
                l_end_search = []
                # The generated messages need to be retrieved
                # since the number of message of prompts is already known
                # search with regex from the next index

                # only one \n because sometimes when generated 
                # the separation is only one new line
                matches = re.finditer("\n[1-9][0-9]?.", df_generated.iloc[i_row,0])

                for match in matches:
                    l_start_search.append(match.start())
                    l_end_search.append(match.end())
               
                # Discarding the initial messages because they are prompt
                l_start_search_subset = l_start_search[number_of_messages_per_prompt:]
                l_end_search_subset = l_end_search[number_of_messages_per_prompt:]

                # discarding the last message because it may be not complete
                for i_regex in range(len(l_start_search_subset) - 1):
                    # Check if the message is empty
                    if l_start_search_subset[i_regex+1] - l_end_search_subset[i_regex] <=1:
                        continue
                    else:
                        # if message is not empty
                        # then extract like this
                        # Remember match start and end give the indices of the start
                        # and end of the substring
                        # ^: start, @: end
                        # \n\n1. Message 1
                        #   ^   @             match1 = text[^:@]         
                        # \n\n2. Message 2
                        #   ^   @              match2 = text[^:@]
                        # \n\n3. Message 3
                        #   ^   @
                        # The idea is to extract from @+1 to the next ^
                        
                        # For the last character of sentence we need to verify that:
                        # if the last character is \n
                        #     reduced message by one character to deleter \n
                        # else:
                        #     nothing

                        # if last character is ":"/colon
                        #     extract until the penultimate sentence
                        #     return/ get out of for loop
                        # else:
                        #     store message and continue for loop
                        
                        extracted_message = df_generated.iloc[i_row,0][l_end_search_subset[i_regex]+1:
                                                                       l_start_search_subset[i_regex+1]]
                       
                        if extracted_message[-1] == "\n":
                            extracted_message = extracted_message[:-1]
                        
                        if extracted_message[-1] == ":":
                            # create a message until the end
                            # check last character is new line or not
                            if df_generated.iloc[i_row,0][l_start_search_subset[len(l_start_search_subset)-1]] == "\n":
                                extracted_message = df_generated.iloc[i_row,0][l_end_search_subset[i_regex]+1:
                                                                                      l_start_search_subset[len(l_start_search_subset)-1]-1]
                            else:
                                extracted_message = df_generated.iloc[i_row,0][l_end_search_subset[i_regex]+1:
                                                                                      l_start_search_subset[len(l_start_search_subset)-1]]
                            l_messages.append(extracted_message)
                            l_index.append(i_row)
                            l_message_position.append(i_regex) 
                            break                           
                        else:
                            l_messages.append(extracted_message)
                            l_index.append(i_row)
                            l_message_position.append(i_regex)

                print("Hello World")
        elif "v4" in filename:

            for i_row in range(df_generated.shape[0]):
                # Find the position of prompt
                # x1 = re.search("Write messages like the previous ones:", df_generated.iloc[i_row,0])
                matches = re.finditer("Write messages like the previous ones:", df_generated.iloc[i_row,0])

                l_match_start = []
                l_match_end = []

                for match in matches:
                    l_match_start.append(match.start())
                    l_match_end.append(match.end())

                # Generate subset after prompt
                if len(l_match_start) == 1:
                    output_subset = df_generated.iloc[i_row,0][l_match_end[0]:]
                # if "Write messages like the previous ones:", it tends to repeat the second time
                # therefore the second one is removed
                else:
                    output_subset = df_generated.iloc[i_row,0][l_match_end[0]:l_match_start[1]]            
                  
                l_start_search = []
                l_end_search = []

                matches = re.finditer("\n[1-9][0-9]?.", output_subset)

                for match in matches:
                    l_start_search.append(match.start())
                    l_end_search.append(match.end())
               
                for i_regex in range(len(l_start_search)):
                    # Check if the message is empty
                    if  i_regex == len(l_start_search) -1:
                        extracted_message = output_subset[l_end_search[i_regex]+1:]
                        
                        if extracted_message[-1] == "\n":
                            extracted_message = extracted_message[:-1]
                            
                        l_messages.append(extracted_message)
                        l_index.append(i_row)
                        l_message_position.append(i_regex)
                    elif l_start_search[i_regex+1] - l_end_search[i_regex] <=1:
                        continue
                    else:
                        # if message is not empty
                        # then extract like this
                        # Remember match start and end give the indices of the start
                        # and end of the substring
                        # ^: start, @: end
                        # \n\n1. Message 1
                        #   ^   @             match1 = text[^:@]         
                        # \n\n2. Message 2
                        #   ^   @              match2 = text[^:@]
                        # \n\n3. Message 3
                        #   ^   @
                        # The idea is to extract from @+1 to the next ^
                        
                        # For the last character of sentence we need to verify that:
                        # if the last character is \n
                        #     reduced message by one character to deleter \n
                        # else:
                        #     nothing

                        # if last character is ":"/colon
                        #     extract until the penultimate sentence
                        #     return/ get out of for loop
                        # else:
                        #     store message and continue for loop
                        
                        extracted_message = output_subset[l_end_search[i_regex]+1:
                                                                       l_start_search[i_regex+1]]
                       
                        if extracted_message[-1] == "\n":
                            extracted_message = extracted_message[:-1]
                        
                        if extracted_message[-1] == ":":
                            # create a message until the end
                            # check last character is new line or not
                            if df_generated.iloc[i_row,0][l_start_search[len(l_start_search)-1]] == "\n":
                                extracted_message = output_subset[l_end_search[i_regex]+1:
                                                                               l_start_search[len(l_start_search)-1]-1]
                            else:
                                extracted_message = output_subset[l_end_search[i_regex]+1:
                                                                               l_start_search[len(l_start_search)-1]]
                            l_messages.append(extracted_message)
                            l_index.append(i_row)
                            l_message_position.append(i_regex) 
                            break                           
                        else:
                            l_messages.append(extracted_message)
                            l_index.append(i_row)
                            l_message_position.append(i_regex)

                print("Hello World")

        else:
            raise ValueError("Prompt version not implemented.")

        # Store as pandas dataframe
        filename = os.path.basename(csv_filepath)

        df_temp = pd.DataFrame( {"Message":l_messages, "index_prompt":l_index, "message_position":l_message_position})

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
