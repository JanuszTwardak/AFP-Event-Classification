from pickle import FALSE
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

        model = model.fit(dataset)
    return model


def main() -> None:

    dataset_name = "331020_afp_newhits"
    preprocessed_input_path = "preprocessed_data/" + dataset_name + ".pkl"
    dataset_test = load_preprocessed_data(preprocessed_input_path)

    preprocessed_df_names = ["331020_afp_newhits", "336505_afp_newhits"]

    model = train_model(preprocessed_df_names)

    scores = pd.DataFrame(model.predict(dataset_test))

    dataset_test = load_preprocessed_data(preprocessed_input_path, drop_evN=False)
    scores.set_index(dataset_test["evN"], inplace=True)

    print(scores.head())
    anomaly_threshold = 0.6

    visualizer = Visualize(
        input_dataset=dataset_name,
        scores=scores,
        anomaly_threshold=anomaly_threshold,
        should_save=False,
        should_show=True,
        root_name=dataset_name,
    )

    visualizer.draw_examples(examples_number=10)

    # pickle.dump(model, open("output/", ROOT_FILE_NAME, "/model.pck"))
    # pickle.dump(scores, open("output/", ROOT_FILE_NAME, "/scores.pck"))


if __name__ == "__main__":
    main()
