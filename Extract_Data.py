from ipaddress import ip_address
import os
from typing import List, Optional
from dask.distributed import Client
import dask.dataframe as dd
from Visualize import Visualize as Vs
import parameters as parameters
import numpy as np
import datashader as ds
import colorcet as cc
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import logging
from colorama import Fore, Back, Style


def save_df_as_ddf(save_path: str, ip_address: str, n_partitions: Optional[int] = 50):
    client = Client(ip_address)

    dataset = dd.read_parquet("preprocessed_data/dask_final", engine="pyarrow")
    names = ["score", "evN", "run_number"]
    scores = dd.read_csv("preprocessed_data/dask_final/scores.csv", names=names)

    combined = dd.merge(dataset, scores)
    combined.repartition(npartitions=n_partitions)
    combined["evN"] = combined["evN"].astype(str)
    combined["run_number"] = combined["run_number"].astype(str)
    combined["scores"] = combined["score"]
    combined = combined.set_index("score", npartitions=n_partitions, drop=True)

    logging.debug("Saving ddf...")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    combined.to_parquet(path=save_path, engine="pyarrow")
    logging.debug("Ddf saved!")


def load_dataset(path: str, npartitions: int, ip_address: str) -> dd:

    client = Client(ip_address)
    logging.debug("Loading ddf dataset...")
    dataset = dd.read_parquet(path)
    dataset.repartition(npartitions=npartitions)
    logging.debug("Ddf dataset loaded!")

    return dataset


def draw_score_boxplot(
    scores: dd,
    flierprops: dict,
    save_dict_path: str,
    file_name: Optional[str] = "score_boxplot",
):

    logging.info("Creating plot...")
    sns.boxplot(x=scores, flierprops=flierprops)

    logging.info("Saving plot...")
    plt.title("Scores boxplot for all events")
    quantiles = np.quantile(scores, np.array([0.00, 0.25, 0.50, 0.75, 1.00]))
    whisker = (quantiles[3] - quantiles[1]) * 1.5 + quantiles[3]
    quantiles = np.append(quantiles, [whisker])
    plt.vlines(
        quantiles,
        [0] * quantiles.size,
        [1] * quantiles.size,
        color="b",
        ls=":",
        lw=0.5,
        zorder=0,
    )
    plt.xticks(np.round(quantiles, 2))

    save_path = os.path.join(save_dict_path, file_name)
    plt.savefig(save_path)
    logging.info("Plot saved")


class Parameters:
    transform_df_to_ddf = False

    ip_address = "tcp://192.168.55.222:8786"
    ddf_path = "results_test_big/dask_merged3/"
    logg_level = logging.DEBUG
    scores_path = "preprocessed_data/dask_final/scores.csv"
    plots_path = "plots"
    n_partitions = 50

    class boxplot:
        flierprops = dict(
            markerfacecolor="0.75", markersize=1, linestyle="none", alpha=0.002
        )


def draw_hits_quantile_boxplots(
    scores_n_hits,
    hits_n,
    flierprops: dict,
    save_dict_path: str,
    file_name: Optional[str] = "hits_quantile_boxplot",
):

    scores_n_hits = scores_n_hits.compute()
    quartiles = scores_n_hits["scores"].quantile(q=[0.0, 0.25, 0.5, 0.75, 1.0])
    quartiles = quartiles.to_numpy()

    print(quartiles, type(quartiles))

    df = pd.DataFrame()

    for i in range(1, 5):
        quarter = scores_n_hits[
            (scores_n_hits["scores"] > quartiles[i - 1])
            & (scores_n_hits["scores"] < quartiles[i])
        ]
        quarter = quarter.reset_index()
        df[f"{i}"] = quarter["hits_n"]

    df = df.melt()
    print(df.head())

    logging.info("Creating plot...")
    sns.boxplot(x="variable", y="value", data=df, flierprops=flierprops)
    logging.info("Saving plot...")
    plt.title("Hits number between quartiles")
    save_path = os.path.join(save_dict_path, file_name)
    plt.savefig(save_path)

    logging.info("Plot saved")


def main() -> None:
    params = Parameters()
    if params.transform_df_to_ddf:
        save_df_as_ddf(save_path=params.ddf_path)

    logging.basicConfig(level=params.logg_level)

    dataset = load_dataset(
        path=params.ddf_path,
        npartitions=params.n_partitions,
        ip_address=params.ip_address,
    )

    if not os.path.isdir(params.plots_path):
        os.mkdir(params.plots_path)

    # draw_score_boxplot(
    #     scores=dataset["scores"],
    #     save_dict_path=params.plots_path,
    #     flierprops=params.boxplot.flierprops,
    # )
    scores_n_hits = dataset[["scores", "hits_n"]]

    draw_hits_quantile_boxplots(
        scores_n_hits=scores_n_hits,
        hits_n=dataset["hits_n"],
        save_dict_path=params.plots_path,
        flierprops=params.boxplot.flierprops,
    )


if __name__ == "__main__":
    main()
