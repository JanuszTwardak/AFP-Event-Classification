from pickle import FALSE
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sympy import true
import uproot
from matplotlib import pyplot as plt
import sys
from isotree import IsolationForest


sys.path.append(".")
from Visualize import Visualize

ROOT_FILE_NAME = "331020_afp_newhits"
PREPROCESSED_INPUT_PATH = "preprocessed_data/" + ROOT_FILE_NAME + ".pkl"


def load_preprocessed_data(PREPROCESSED_INPUT_PATH):
    df = pd.read_pickle(PREPROCESSED_INPUT_PATH)
    print(df.head())
    return df


def plot_space(Z, space_index, X):
    df = pd.DataFrame({"z": Z}, index=space_index)
    df = df.unstack()
    df = df[df.columns.values[::-1]]
    plt.imshow(df, extent=[-3, 3, -3, 3], cmap="hot_r")
    plt.scatter(x=X["x"], y=X["y"], alpha=0.15, c="navy")


# def planes_preview(visualizer, events_number_to_show=10):
#     normal_counter = 0
#     anomaly_counter = 0

#     visualizer.draw_planes(1344530188, path="output/331020_afp_newhits/")

#     for i, row in loaded_data.iterrows():
#         if row["score"] > 0.7 and anomaly_counter < 10:
#             visualizer.draw_planes(
#                 1344989546, path="output/331020_afp_newhits/anomaly/"
#             )
#             anomaly_counter = anomaly_counter + 1
#         elif normal_counter < 10:
#             visualizer.draw_planes(1344989546, path="output/331020_afp_newhits/normal/")
#             normal_counter = normal_counter + 1


def main():

    loaded_data = load_preprocessed_data(PREPROCESSED_INPUT_PATH)
    loaded_data.set_index(loaded_data["evN"], inplace=True)
    dataset = pd.concat(
        [loaded_data["hits_n"], loaded_data["evN"]], axis=1, ignore_index=False
    )

    space = (
        np.array(
            np.meshgrid(
                loaded_data["hits_n"],
                loaded_data["hit_row_1"],
            )
        )
        .reshape((2, -1))
        .T
    )

    space_index = pd.MultiIndex.from_arrays([space[:, 0], space[:, 1]])

    model = IsolationForest(ndim=1, ntrees=100).fit(dataset)
    scores = pd.DataFrame(model.predict(dataset))
    scores.set_index(loaded_data["evN"], inplace=True)
    threshold = 0.6

    # print(loaded_data.sort_values(by="score", ascending=True))

    visualizer = Visualize(
        input_dataset=ROOT_FILE_NAME,
        scores=scores,
        anomaly_threshold=threshold,
        should_save=False,
        should_show=True,
        root_name=ROOT_FILE_NAME,
    )

    # visualizer.draw_planes(event_number=1344053190, side="c")
    visualizer.draw_examples(examples_number=10)


if __name__ == "__main__":
    main()
