import os
from pickle import FALSE
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


def load_preprocessed_data(
    preprocessed_input_path: str, drop_evN: Optional[bool] = True
) -> pd.DataFrame:
    df = pd.read_pickle(preprocessed_input_path)
    df.set_index(df["evN"], inplace=True)
    if drop_evN:
        df.drop("evN", axis=1, inplace=True)
    df.drop("side", axis=1, inplace=True)
    print(df)
    return df


def train_model(preprocessed_df_names: List[str]) -> IsolationForest:

    model = IsolationForest(ndim=1, ntrees=100)

    for filename in preprocessed_df_names:
        preprocessed_input_path = "preprocessed_data/" + filename + ".pkl"
        dataset = load_preprocessed_data(preprocessed_input_path)
        model = model.fit(dataset.drop(columns=["evN", "b"]))
    return model


def save_model(
    model: IsolationForest, path: Optional[str] = None, progress: Optional[float] = 1.0
) -> None:

    if path is None:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        progress_str = f"{str(int(progress*100))}"
        path = os.path.join(
            f"IFE_{time_str}_{progress_str}.model",
        )
        if not os.path.isdir(path):
            os.mkdir(path)

    else:
        if not os.path.isdir(path):
            os.mkdir(path)

    model.to_pickle(path)


def main() -> None:

    dataset_name = "331020_afp_newhits"
    preprocessed_input_path = "preprocessed_data/" + dataset_name + ".pkl"

    preprocessed_df_names = ["331020_afp_newhits", "336505_afp_newhits"]

    model = train_model(preprocessed_df_names)
    save_model(model)

    dataset_test = load_preprocessed_data(preprocessed_input_path, drop_evN=False)
    scores = pd.DataFrame(model.predict(dataset_test))
    scores.set_index(dataset_test["evN"], inplace=True)

    print(dataset_test.head())
    anomaly_threshold = 0.6

    visualizer = Visualize(
        input_dataset=dataset_name,
        scores=scores,
        anomaly_threshold=anomaly_threshold,
        should_save=True,
        should_show=False,
        root_name=dataset_name,
    )


if __name__ == "__main__":
    main()
