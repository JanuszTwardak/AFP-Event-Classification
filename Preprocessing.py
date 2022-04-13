from sklearn.model_selection import StratifiedShuffleSplit
import uproot
import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Mapping
import sys

sys.path.append(".")


def read_root(ROOT_FILE_NAME, BRANCHES, ENTRY_LIMIT=None):
    print("Loading and converting root file... ", end="")
    ROOT_INPUT_PATH = "input_root/" + ROOT_FILE_NAME + ".root"
    root_file = uproot.open(ROOT_INPUT_PATH)
    print("OPENED")
    tree = root_file["TreeHits"]
    dataset = tree.arrays(BRANCHES, library="pd", entry_stop=ENTRY_LIMIT)
    print("EXTRACTED")
    dataset = dataset.copy()
    print("DONE!")
    return dataset


def preprocess_single_dataframe(dataset, functions):
    print("Loading and converting root file... ", end="")
    for step in functions:
        dataset = step(dataset)
    print("DONE!")
    return dataset


def add_hits_number(dataset):
    dataset["a_hits_n"] = dataset.filter(regex="^hits\\[[01]", axis=1).sum(axis=1)
    dataset["c_hits_n"] = dataset.filter(regex="^hits\\[[23]", axis=1).sum(axis=1)
    dataset.drop(dataset.filter(regex="^hits\\[", axis=1), axis=1, inplace=True)
    return dataset


def add_average_coordinates(dataset):

    # TODO rewrite - ugly as hell
    # first detector (in hit order, which means for side A we take detector #2 -> detector #1 data)
    weights_a_1 = dataset.filter(regex="^hits_q\\[1", axis=1).where(
        dataset.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
    )
    weights_c_1 = dataset.filter(regex="^hits_q\\[2", axis=1).where(
        dataset.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
    )
    rows_a = dataset.filter(regex="^hits_row\\[1", axis=1)
    rows_c = dataset.filter(regex="^hits_row\\[2", axis=1)
    dataset["a_hit_row_1"] = (rows_a * weights_a_1.values).sum(
        axis=1
    ) / weights_a_1.sum(axis=1)
    dataset["c_hit_row_1"] = (rows_c * weights_c_1.values).sum(
        axis=1
    ) / weights_c_1.sum(axis=1)

    # second detector (in hit order)
    weights_a_2 = dataset.filter(regex="^hits_q\\[0", axis=1).where(
        dataset.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
    )
    weights_c_2 = dataset.filter(regex="^hits_q\\[3", axis=1).where(
        dataset.filter(regex="^hits_q\[", axis=1) != -1000001.0, 0
    )
    dataset.drop(dataset.filter(regex="^hits_q", axis=1), axis=1, inplace=True)
    rows_a = dataset.filter(regex="^hits_row\\[0", axis=1)
    rows_c = dataset.filter(regex="^hits_row\\[3", axis=1)
    dataset["a_hit_row_2"] = (rows_a * weights_a_2.values).sum(
        axis=1
    ) / weights_a_2.sum(axis=1)
    dataset["c_hit_row_2"] = (rows_c * weights_c_2.values).sum(
        axis=1
    ) / weights_c_2.sum(axis=1)

    del [rows_a, rows_c]
    gc.collect()

    columns_a = dataset.filter(regex="^hits_col\\[1", axis=1)
    columns_c = dataset.filter(regex="^hits_col\\[2", axis=1)
    dataset["a_hit_column_1"] = (columns_a * weights_a_1.values).sum(
        axis=1
    ) / weights_a_1.sum(axis=1)
    dataset["c_hit_column_1"] = (columns_c * weights_c_1.values).sum(
        axis=1
    ) / weights_c_1.sum(axis=1)
    columns_a = dataset.filter(regex="^hits_col\\[0", axis=1)
    columns_c = dataset.filter(regex="^hits_col\\[3", axis=1)
    dataset["a_hit_column_2"] = (columns_a * weights_a_2.values).sum(
        axis=1
    ) / weights_a_2.sum(axis=1)
    dataset["c_hit_column_2"] = (columns_c * weights_c_2.values).sum(
        axis=1
    ) / weights_c_2.sum(axis=1)

    del [columns_a, columns_c]
    del [weights_a_1, weights_c_1, weights_a_2, weights_c_2]
    gc.collect()

    return dataset


def merge_detector_sides(dataset: pd.DataFrame) -> pd.DataFrame:
    from math import sqrt

    MINIMUM_HIT_NUMBER = 1
    MAXIMUM_HIT_NUMBER = 100

    # TODO reformat code below
    buffor = dataset.drop(dataset.filter(regex="^c", axis=1), axis=1, inplace=False)
    buffor.rename(
        columns={
            "a_hits_n": "hits_n",
            "a_hit_row_1": "hit_row_1",
            "a_hit_row_2": "hit_row_2",
            "a_hit_column_1": "hit_column_1",
            "a_hit_column_2": "hit_column_2",
            "a_std_col": "_std_col",
            "a_std_row": "_std_row",
        },
        inplace=True,
    )
    buffor["side"] = "a"

    dataset.drop(dataset.filter(regex="^a", axis=1), axis=1, inplace=True)
    dataset.rename(
        columns={
            "c_hits_n": "hits_n",
            "c_hit_row_1": "hit_row_1",
            "c_hit_row_2": "hit_row_2",
            "c_hit_column_1": "hit_column_1",
            "c_hit_column_2": "hit_column_2",
            "c_std_col": "_std_col",
            "c_std_row": "_std_row",
        },
        inplace=True,
    )
    dataset["side"] = "c"
    dataset = dataset.append(buffor)

    dataset = dataset[dataset["hits_n"] >= MINIMUM_HIT_NUMBER]
    dataset = dataset[dataset["hits_n"] <= MAXIMUM_HIT_NUMBER]

    return dataset


def add_hit_std_deviation(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["a_std_col"] = dataset.filter(regex="^hits_col\\[[01]", axis=1).std(axis=1)
    dataset["a_std_row"] = dataset.filter(regex="^hits_row\\[[01]", axis=1).std(axis=1)

    dataset["c_std_col"] = dataset.filter(regex="^hits_col\\[[23]", axis=1).std(axis=1)
    dataset["c_std_row"] = dataset.filter(regex="^hits_row\\[[23]", axis=1).std(axis=1)

    dataset.drop(dataset.filter(regex="^hits_col", axis=1), axis=1, inplace=True)
    dataset.drop(dataset.filter(regex="^hits_row", axis=1), axis=1, inplace=True)

    return dataset


def merge_std_deviations(dataset: pd.DataFrame) -> pd.DataFrame:

    dataset["std_distance"] = (
        dataset["_std_col"] * dataset["_std_col"].values
        + dataset["_std_row"].values * dataset["_std_row"].values
    )
    dataset["std_distance"] = dataset["std_distance"].pow(1 / 2)

    dataset.drop(dataset.filter(regex="^_", axis=1), axis=1, inplace=True)

    return dataset


def scale_min_max(dataset: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    buffor_evn = dataset["evN"]
    buffor_side = dataset["side"]

    dataset.drop("evN", axis=1, inplace=True)
    dataset.drop("side", axis=1, inplace=True)

    scaled_values = scaler.fit_transform(dataset)
    dataset.loc[:, :] = scaled_values
    dataset = dataset.join([buffor_evn, buffor_side])

    # change order
    cols = list(dataset.columns.values)
    cols = cols[-2:] + cols[:-2]
    dataset = dataset[cols]
    return dataset


def set_indexes(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset.set_index(dataset["evN"], inplace=True)
    print(dataset.head())
    return dataset


def save_preprocessed_dataframe(
    dataset: pd.DataFrame, file_name: str, path: Optional[str] = None
):
    print("Saving dataset... ", end="")
    if path is None:
        path = "preprocessed_data/" + file_name + ".pkl"
        dataset.to_pickle(path)
    else:
        dataset.to_pickle(path)
    print("DONE!")


def preprocess_all(
    root_files: Mapping[str, int],
    branches: Optional[List[str]] = ["evN", "hits", "hits_row", "hits_col", "hits_q"],
):

    for file_name in root_files:

        preprocess_functions = [
            add_hits_number,
            add_average_coordinates,
            add_hit_std_deviation,
            merge_detector_sides,
            merge_std_deviations,
            scale_min_max,
            set_indexes,
        ]

        entry_limit = root_files.get(file_name)

        dataset = read_root(file_name, branches, entry_limit)
        dataset = preprocess_single_dataframe(dataset, preprocess_functions)
        save_preprocessed_dataframe(dataset, file_name=file_name)

        print("--- Preprocessing finished! ---")
        print("Shape of the final dataset: ", dataset.shape)


def main():
    root_files = {"331020_afp_newhits": None, "336505_afp_newhits": 1000}
    preprocess_all(root_files)


if __name__ == "__main__":
    main()
