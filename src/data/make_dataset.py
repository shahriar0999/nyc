# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

import sys
import yaml
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(data_path):
    """
    Loads the dataset from the given path.
    Args:
        data_path (str): The path to the dataset.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(data_path)
    return df

def split_dataset(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the train and test sets to the output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))['make_dataset']
    
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'

    data = load_dataset(data_path)
    train_data, test_data = split_dataset(data, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path)

if __name__ == '__main__':
    main()