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

# Inspiration:
# https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py
# https://blog.paperspace.com/automated-metrics-for-evaluating-generated-text/

# https://github.com/geek-ai/Texygen/blob/master/docs/evaluation.md#self-bleu-score
# Self-BLEU score
# We propose Self-BLEU, a metric to evaluate the diversity of the generated data. Since BLEU aims to assess how similar two sentences are, it can also be used to evaluate how one sentence resembles the rest in a generated collection. Regarding one sentence as hypothesis and the others as reference, we can calculate BLEU score for every generated sentence, and define the average BLEU score to be the Self-BLEU of the document.

def get_reference(filepath):
    
    l_tokenized = []
    
    df_messages = pd.read_csv(filepath, index_col = 0)  
    l_messages = df_messages["Message"].to_list()
    
    for message in l_messages:
        l_tokenized.append(word_tokenize(message))
    
    return l_tokenized

def get_selfbleu_score(filepath, config_json):
    nltk.download('punkt')
    reference_sentences = get_reference(filepath)
       
    n_grams = config_json["n_grams"]
    weight = [1. / n_grams for _ in range(n_grams)]
    
    l_bleu = []
    
    for i, hypothesis_sentence in enumerate(reference_sentences):
        print(i,"out of",len(reference_sentences))
        remaining_sentences = copy.deepcopy(reference_sentences)
        remaining_sentences.remove(hypothesis_sentence)
        
        bleu_temp = sentence_bleu(remaining_sentences, hypothesis_sentence, weight,
                                    smoothing_function=SmoothingFunction().method1)
        print(bleu_temp)
        l_bleu.append(bleu_temp)
        
    return sum(l_bleu) / len(l_bleu)

def get_selfbleu_single(reference_sentences, hypothesis_sentence, weight):

    remaining_sentences = copy.deepcopy(reference_sentences)
    remaining_sentences.remove(hypothesis_sentence)
        
    return sentence_bleu(remaining_sentences, hypothesis_sentence, weight,
                                    smoothing_function=SmoothingFunction().method1)

def get_selfbleu_score_parallel(filepath, config_json):
    reference_sentences = get_reference(filepath)
    num_cores = get_n_parallel_processes(config_json)
    print("num_cores =", num_cores)
    n_grams = config_json["n_grams"]
    weight = [1. / n_grams for _ in range(n_grams)]
    
    results_selfbleu = Parallel(n_jobs=num_cores)( delayed(get_selfbleu_single)(reference_sentences, hypothesis_sentence, weight) \
                                        for hypothesis_sentence in reference_sentences )

    return np.mean(results_selfbleu), np.std(results_selfbleu, ddof=1)

def get_df_bleu(config_json):
    os.makedirs(config_json["path_output"], exist_ok = True)

    df_all = pd.DataFrame(columns = ["filename","self-bleu","std dev"])

    for filepath in config_json["list_filenames"]:
        selfbleu_value, std_dev = get_selfbleu_score_parallel(filepath, config_json)    
        filename = os.path.basename(filepath)
        df_temp = pd.DataFrame({"filename":[filename], 
                                "self-bleu":[selfbleu_value],
                                "std dev":[std_dev]})

        df_all = pd.concat([df_all, df_temp], ignore_index=True)

    path_output_file = os.path.join(config_json["path_output"],
                                        f"Self-BLEU_{config_json['n_grams']}.csv")
    df_all.to_csv(path_output_file)

def main():
    args = get_parser()
    config_json = get_json(args)
    
    # config_json["list_filenames"][0]
    # bleu_temp =get_selfbleu_score_parallel(config_json["list_filenames"][0], config_json)
    
    get_df_bleu(config_json)
    

if __name__ == "__main__":
    main()