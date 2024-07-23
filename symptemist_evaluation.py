"""
SympTEMIST evaluation library main script.
Heavily based upon the MedProcNER/ProcTEMIST evaluation library (https://github.com/TeMU-BSC/medprocner_evaluation_library)
@author: salva
"""

import os

import pandas as pd

from datetime import datetime
from argparse import ArgumentParser

import utils


def main(argv=None):
    """
    Parse options and call the appropriate evaluation scripts
    """
    # Parse options
    parser = ArgumentParser()
    parser.add_argument("-r", "--reference", dest="reference",
                      help=".TSV file with Gold Standard or reference annotations", required=True)
    parser.add_argument("-p", "--prediction", dest="prediction",
                      help=".TSV file with your predictions", required=True)
    parser.add_argument("-t", "--task", dest="task", choices=['ner', 'norm', 'multi'],
                      help="Task that you want to evaluate (ner, norm or multi)", required=True)
    parser.add_argument("-o", "--output", dest="output",
                      help="Path to save the scoring results", required=True)
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                      help="Set to True to print the results for each individual file instead of just the final score")
    args = parser.parse_args(argv)

    # Set output file name
    timedate = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = args.prediction.split('/')[-1][:-4]
    out_file = os.path.join(args.output, 'symptemist_results_{}_{}_{}.txt'.format(args.task, fname, timedate))

    # Read gold_standard and predictions
    print("Reading reference and prediction .tsv files")
    df_gs = pd.read_csv(args.reference, sep="\t", engine='python', quoting=3, dtype=str)
    df_preds = pd.read_csv(args.prediction, sep="\t", engine='python', quoting=3, dtype=str)
    if args.task in ['ner', 'norm']:
        df_preds = df_preds.drop_duplicates(
            subset=["filename", "label", "start_span", "end_span"]).reset_index(drop=True)  # Remove any duplicate predictions

    if args.task == "ner":
        calculate_ner(df_gs, df_preds, out_file, args.verbose)
    elif args.task == "norm":
        calculate_norm(df_gs, df_preds, out_file, args.verbose)
    elif args.task == "multi":
        calculate_multi(df_gs, df_preds, out_file, args.verbose)
    else:
        print('Please choose a valid task (ner, norm, multi)')


def calculate_ner(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 1 (ner)")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text",  "label"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label"]].values.tolist()).to_list()
    scores = utils.calculate_scores(list_gs_per_doc, list_preds_per_doc, 'ner')
    utils.write_results('ner', scores, output_path, verbose)


def calculate_norm(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 2 (norm)")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    scores = utils.calculate_scores(list_gs_per_doc, list_preds_per_doc, 'norm')
    utils.write_results('norm', scores, output_path, verbose)


def calculate_multi(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 3 (multilingual)")
    # Group annotations by filename
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[[
        "filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(
        lambda x: x[["filename", 'start_span', 'end_span', "text", "label", "code"]].values.tolist()).to_list()
    scores = utils.calculate_scores(list_gs_per_doc, list_preds_per_doc, 'multi')
    utils.write_results('multi', scores, output_path, verbose)


if __name__ == "__main__":
    main()