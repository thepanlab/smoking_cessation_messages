''' process_LIWC_results.py
    It allows to estimate mean and std error for some LIWC metrics

    python process_LIWC_results.py -j ./process_LIWC_results_vB/process_LIWC_results_v1_vB.json
'''


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

def get_df_summary(config_json):
    
    file_path = config_json["LIWC_file"]
    
    df_LIWC = pd.read_csv(file_path)
    
    l_column_features = config_json["column_features"]
    
    l_column_df = ["type"]
    l_column_df.extend([ f"{feature}-{suffix}" for feature in l_column_features for suffix in ["mean","std_err"]])
    
    a_unique_type = df_LIWC["type"].unique()
    
    df_LIWC_summary = pd.DataFrame(columns = l_column_df)
    
    for type in a_unique_type:
        dict_row = {}

        dict_row["type"] = type

        for feature in l_column_features:    
            mean_temp = df_LIWC[df_LIWC["type"] == type][feature].mean()
            std_temp = df_LIWC[df_LIWC["type"] == type][feature].std(ddof=1)
            count = df_LIWC[df_LIWC["type"] == type][feature].count()
            
            std_err = std_temp/(count**0.5)

            # df_LIWC_summary = df_LIWC_summary.append({"type":type, f"{feature}-mean" : mean_temp,
            #                                           f"{feature}-std_err" : std_temp}, ignore_index = True)

            # df_LIWC_summary = pd.concat([df_LIWC_summary,df_temp], ignore_index=True)
            dict_row[f"{feature}-mean"] = mean_temp
            dict_row[f"{feature}-std_err"] = std_err

        df_temp =pd.DataFrame(dict_row, index=[0])

        df_LIWC_summary = pd.concat([df_LIWC_summary, df_temp], ignore_index=True)

        os.makedirs(config_json["path_directory_output"], exist_ok = True)

    path_output_file = os.path.join(config_json["path_directory_output"], f"LIWC_summary_{config_json['suffix_file']}.csv")
    df_LIWC_summary.to_csv(path_output_file, encoding = "utf-8")

        # df_LIWC_summary = df_LIWC_summary.merge(df_temp, on="type", how = "outer")

    print("Hello World")

def main():
    args = get_parser()
    config_json = get_json(args)

    get_df_summary(config_json)

if __name__ == "__main__":
    main()