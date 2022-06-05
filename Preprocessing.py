from codecs import ignore_errors
import uproot
import os
import gc
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Mapping
from warnings import simplefilter
import parameters as parameters


class Preprocessing:
    def read_full_root(root_name, BRANCHES, ENTRY_LIMIT=None):
        print("Loading and converting root file... ", end="")
        root_path = "input_root/" + root_name + ".root"
        with uproot.open(root_path) as root_file:
            tree = root_file["TreeHits"]

        dataset = tree.arrays(BRANCHES, library="pd", entry_stop=ENTRY_LIMIT)
        dataset = dataset.copy()
        return dataset

    def preprocess_single_dataframe(
        dataset: pd.DataFrame, functions: callable, root_name: str
    ) -> pd.DataFrame:

        for step in functions:
            dataset = step(dataset)

        dataset["run_number"] = int(root_name[0:6])
        dataset["run_number"] = dataset["run_number"].astype("category")

        return dataset

    def add_hits_number(dataset):

        dataset["a_hits_n"] = dataset.filter(regex="^hits\\[[01]", axis=1).sum(axis=1)
        dataset["c_hits_n"] = dataset.filter(regex="^hits\\[[23]", axis=1).sum(axis=1)
        dataset.drop(dataset.filter(regex="^hits\\[", axis=1), axis=1, inplace=True)
        return dataset

    def add_average_coordinates(dataset):

        # TODO rewrite
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

        dataset[["a_hit_row_2", "c_hit_row_2"]] = dataset[
            ["a_hit_row_2", "c_hit_row_2"]
        ].apply(pd.to_numeric, downcast="unsigned")

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

    def merge_detector_sides(
        dataset: pd.DataFrame,
        min_hits: Optional[int] = 1,
        max_hits: Optional[int] = 100,
    ) -> pd.DataFrame:
        from math import sqrt

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
        dataset["side"] = dataset["side"].astype("category")

        dataset = dataset[dataset["hits_n"] >= min_hits]
        dataset = dataset[dataset["hits_n"] <= max_hits]

        return dataset

    def add_hit_std_deviation(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["a_std_col"] = dataset.filter(regex="^hits_col\\[[01]", axis=1).std(
            axis=1
        )
        dataset["a_std_row"] = dataset.filter(regex="^hits_row\\[[01]", axis=1).std(
            axis=1
        )

        dataset["c_std_col"] = dataset.filter(regex="^hits_col\\[[23]", axis=1).std(
            axis=1
        )
        dataset["c_std_row"] = dataset.filter(regex="^hits_row\\[[23]", axis=1).std(
            axis=1
        )

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

    def optimize_memory(dataset: pd.DataFrame) -> pd.DataFrame:
        floats = list(dataset.drop(columns=["evN", "side", "hits_n"]).columns)
        ints = ["hits_n"]
        categories = ["evN", "side"]

        dataset[floats] = dataset[floats].astype("float16")
        dataset[ints] = dataset[ints].astype("uint8")
        dataset[categories] = dataset[categories].astype("category")

        return dataset

    def scale_min_max(dataset: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        evN = dataset["evN"]
        side = dataset["side"]

        dataset.drop(["evN", "side", "run_number"], axis=1, inplace=True)

        scaled_values = scaler.fit_transform(dataset)
        dataset.loc[:, :] = scaled_values
        dataset = dataset.join([evN, side])

        # change order
        cols = list(dataset.columns.values)
        cols = cols[-2:] + cols[:-2]
        dataset = dataset[cols]
        return dataset

    def set_indexes(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.set_index(dataset["evN"], inplace=True)
        return dataset

    def save_preprocessed_dataframe(
        dataset: pd.DataFrame, file_name: str, path: Optional[str] = None
    ):
        if path is None:
            path = os.path.join(parameters.preprocessed_data_path, file_name + ".pkl")
            dataset.to_pickle(path)
        else:
            dataset.to_pickle(path)

    @staticmethod
    def preprocess_all(
        root_files: List[str],
        chunk_size: str,
        branches: List[str],
    ):

        preprocess_functions = []

        for function in parameters.preprocess_functions:
            preprocess_functions.append(getattr(Preprocessing, function))

        for single_root_name in root_files:
            root_path = str(
                os.path.join(parameters.path_to_root_dict, single_root_name + ".root")
            )

            with uproot.open(root_path) as file:
                total_size = float(os.path.getsize(root_path)) * 1e-9
                tree = file["TreeHits"]

                chunk_iter = 0

                for chunk in tree.iterate(branches, library="pd", step_size=chunk_size):
                    chunk_iter += 1
                    file_name = single_root_name + str(chunk_iter)

                    chunk = Preprocessing.preprocess_single_dataframe(
                        chunk, preprocess_functions, single_root_name
                    )

                    Preprocessing.save_preprocessed_dataframe(
                        chunk, file_name=file_name
                    )

                    size_done = int(chunk_size[:-3]) * chunk_iter * 1e-3

                    print(
                        f"preprocessing: {single_root_name} | progress: {size_done:.2f}/{total_size:.2f} GB"
                    )
                    print(chunk)
                    print("@@ MEMORY USAGE @@", chunk.memory_usage(deep=True))
                    print(chunk.info())
        print("Preprocess finished!")

    @staticmethod
    def ignore_simple_df_warning():
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        sys.path.append(".")


def main():
    Preprocessing.ignore_simple_df_warning()

    Preprocessing.preprocess_all(
        parameters.root_names,
        chunk_size=parameters.memory_chunk_size,
        branches=parameters.preprocess_branches,
    )


if __name__ == "__main__":
    main()
