import os
from pickle import FALSE
import shutil
import time
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sympy import true
import uproot
from typing import List, Optional
from matplotlib import pyplot as plt
import sys
from isotree import IsolationForest
import pickle
from Visualize import Visualize
import pickle
import parameters as parameters
from Preprocessing import Preprocessing


def load_preprocessed_data(preprocessed_input_path: str) -> pd.DataFrame:
    """load_preprocessed_data Loads and returns pickled dataframe file from a given path.

    Parameters
    ----------
    preprocessed_input_path : str
        Filepath to previously pickled dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe loaded fully from single file.
    """
    df = pd.read_pickle(preprocessed_input_path)
    df.set_index(df["evN"], inplace=True)
    df.drop("side", axis=1, inplace=True)
    return df


def train_model(
    preprocessed_df_names: List[str],
    trees_number: int,
    preprocessed_df_dict_path: Optional[List[str]] = parameters.preprocessed_data_path,
) -> IsolationForest:
    """train_model Trains Isolation Forest model on given preprocessed dataframes.

    Parameters
    ----------
    preprocessed_df_names : List[str]
        List of names of training dataframe files.
    preprocessed_df_dict_path : Optional[List[str]], optional
        Path to folder containing training dataframes, by default parameters.preprocessed_data_path

    Returns
    -------
    IsolationForest
        Fully trained Isolation Forest model
    """

    model = IsolationForest(ndim=parameters.trees_dimension, ntrees=trees_number)

    # for df_name in preprocessed_df_names:
    #     df_chunks = [
    #         file
    #         for file in os.listdir(preprocessed_df_dict_path)
    #         if file.startswith(df_name)
    #     ]

    #     for chunk in df_chunks:
    #         preprocessed_input_path = os.path.join(preprocessed_df_dict_path, chunk)
    #         dataset = load_preprocessed_data(preprocessed_input_path)
    #         model = model.partial_fit(dataset.drop(columns=["evN", "run_number"]))

    sampled_df_paths = get_sampled_df_paths(
        preprocessed_df_names, preprocessed_df_dict_path
    )

    for path in sampled_df_paths:
        dataset = load_preprocessed_data(path)
        model = model.partial_fit(dataset.drop(columns=["evN", "run_number"]))

    return model


def get_sampled_df_paths(
    preprocessed_df_names: List[str], preprocessed_df_dict_path: str
) -> List[str]:

    sampled_df_paths = []

    for df_name in preprocessed_df_names:
        sampled_df_paths.extend(
            [
                os.path.join(preprocessed_df_dict_path, file)
                for file in os.listdir(preprocessed_df_dict_path)
                if file.startswith(df_name)
            ]
        )

    return sampled_df_paths


def predict_scores(
    model: IsolationForest,
    preprocessed_df_names: List[str],
    preprocessed_df_dict_path: Optional[str] = parameters.preprocessed_data_path,
    final_results_path: Optional[str] = parameters.final_results_path,
) -> None:
    """predict_scores Using already trained model predict dataframe anomaly scores. They will be saved in .csv file.

    Parameters
    ----------
    model : IsolationForest
        Fully trained IsolationForest model.
    preprocessed_df_names : List[str]
        List of dataframe names (that were used to train model) to predict score of.
    preprocessed_df_dict_path : Optional[List[str]], optional
        Path to folder containing training dataframes, by default parameters.preprocessed_data_path
    final_results_path : Optional[str], optional
        Path to folder containing final results of model data, by default parameters.final_results_path
    """

    save_path = parameters.scores_path

    sampled_df_paths = get_sampled_df_paths(
        preprocessed_df_names, preprocessed_df_dict_path
    )
    sampled_df_number = len(sampled_df_paths)

    if not os.path.isdir(final_results_path):
        os.mkdir(final_results_path)

    with open(save_path, mode="a"):
        pass

    for count, path in enumerate(sampled_df_paths):
        print(
            f"Predicting scores. Progress: {int(sampled_df_paths.index(path)/sampled_df_number*100)}"
        )

        dataset = load_preprocessed_data(path)

        scores = pd.DataFrame(
            model.predict(dataset.drop(columns=["evN", "run_number"])),
            columns=["scores"],
        )

        scores.set_index(dataset["evN"], inplace=True)
        scores["event_number"] = dataset["evN"]
        scores["run_number"] = dataset["run_number"]

        scores = optimize_csv_memory(scores)
        scores.to_csv(save_path, mode="a", index=False, header=False)


def optimize_csv_memory(dataset: pd.DataFrame) -> pd.DataFrame:
    floats = ["scores"]
    categories = ["event_number", "run_number"]

    dataset[floats] = dataset[floats].astype("float16")
    dataset[categories] = dataset[categories].astype("category")

    return dataset


def save_model(
    model: IsolationForest,
    path: Optional[str] = None,
    trained_level: Optional[float] = 1.0,
) -> None:
    """save_model Saves Isolation Forest model using pickle.

    Parameters
    ----------
    model : IsolationForest
        Model to be saved.
    path : str, optional
        Path where model should be saved., by default None
    trained_level : Optional[float], optional
        Level of training progress, in case of saving model multiple times during training process, by default 1.0
    """

    if path is None:

        progress_str = f"{str(int(trained_level*100))}"
        path = os.path.join(parameters.final_results_path, f"IFE_{progress_str}.pkl")

    if not os.path.isdir(parameters.final_results_path):
        os.mkdir(parameters.final_results_path)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_parameters():
    path = parameters.final_results_path
    if not os.path.isdir(path):
        os.mkdir(path)
    shutil.copy("parameters.py", path)
    shutil.copy("parameters.py", os.path.join(path, "parameters.txt"))


def main() -> None:
    """main Trains and predicts anomaly scores of given datasets."""
    Preprocessing.ignore_simple_df_warning()
    save_parameters()

    DIR = parameters.preprocessed_data_path
    trees_number = len(
        [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    )
    preprocessed_df_names = parameters.df_names
    model = train_model(preprocessed_df_names, trees_number)
    save_model(model)
    predict_scores(model, parameters.df_names)


if __name__ == "__main__":
    main()
