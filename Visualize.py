from typing import List, Optional, Tuple, Union
import numpy as np
from matplotlib import pyplot as plt
import uproot
import time
import os
import pandas as pd
from pathlib import Path
import pickle
from math import fsum
import parameters
from dask import dataframe as ddf


class Visualize:
    def __init__(
        self,
        should_save: bool,
        should_show: bool,
        scores_path: str,
        anomaly_threshold: float,
        path_to_root_dict: str,
        save_path: str,
    ) -> None:

        self.should_save = should_save
        self.should_show = should_show
        self.save_path = save_path
        self.path_to_root_dict = path_to_root_dict
        self.scores_path = scores_path
        self.anomaly_threshold = anomaly_threshold

    def load_dataset(self, input_dataset: Union[pd.DataFrame, str]) -> pd.DataFrame:

        return (
            input_dataset
            if isinstance(input_dataset, pd.DataFrame)
            else self.load_root_from_path(input_dataset)
        )

    def load_scores(input_scores: Union[pd.DataFrame, str]) -> pd.DataFrame:

        return (
            input_scores
            if isinstance(input_scores, pd.DataFrame)
            else pd.read_pickle(input_scores)
        )

    def load_root_from_path(root_name: str) -> pd.DataFrame:
        root_path = os.path.join(Visualize.path_to_root_dict, root_name + ".root")

        with uproot.open(root_path) as file:
            tree = file["TreeHits"]

        dataset = tree.arrays(
            ["evN", "hits", "hits_row", "hits_col", "hits_q"],
            library="pd",
        )
        return dataset

    def draw_histogram(self, bins: Optional[int] = 30) -> None:
        plt.hist(self.scores, density=True, bins=bins)
        plt.show()

    def create_image_as_matrix(
        self,
        event: pd.DataFrame,
    ) -> np.ndarray:

        DETECTORS_NUMBER = 4
        PLANES_PER_DETECTOR = 4
        Y_PIXELS = 336
        X_PIXELS = 80

        image_matrix = np.zeros(
            shape=(DETECTORS_NUMBER, PLANES_PER_DETECTOR, Y_PIXELS, X_PIXELS)
        )

        regex_row = "^hits_row\\[X]\\[X"
        regex_col = "^hits_col\\[X]\\[X"

        for detector in range(4):
            regex_col = regex_col.replace(regex_col[len(regex_col) - 5], str(detector))
            regex_row = regex_row.replace(regex_row[len(regex_row) - 5], str(detector))

            for plane in range(4):
                regex_col = regex_col.replace(regex_col[len(regex_col) - 1], str(plane))
                regex_row = regex_row.replace(regex_row[len(regex_row) - 1], str(plane))

                hit_rows = event.filter(regex=regex_row, axis=1)
                hit_columns = event.filter(regex=regex_col, axis=1)
                hit_rows = hit_rows.transpose()
                hit_columns = hit_columns.transpose()

                hit_rows.set_axis(["coordinate"], axis=1, inplace=True)
                hit_columns.set_axis(["coordinate"], axis=1, inplace=True)

                for index, y in enumerate(hit_rows["coordinate"]):
                    if y != -1:
                        x = hit_columns["coordinate"].iloc[index]
                        if x >= 80:
                            x = 79
                        if y >= 336:
                            y = 335
                        charge_value = 1
                        image_matrix[detector, plane, y, x] = charge_value

        return image_matrix

    def plot_picture(
        self,
        fig: plt.figure,
        image: np.array,
        side: str,
        event_number: int,
        run_number: int,
        event_score: float,
        should_save: Optional[bool] = True,
        should_show: Optional[bool] = False,
    ) -> plt.figure():

        rows = 2 if side == "all" else 1
        columns = 8
        subplot_position = 1
        detectors = {
            side == "all": {0, 1, 2, 3},
            side == "a": {0, 1},
            side == "c": {2, 3},
        }[True]

        for detector in detectors:
            for plane in range(4):
                fig.add_subplot(rows, columns, subplot_position)
                subplot_position += 1
                # TODO plane and detector should be swapped, probably some error?
                plt.axis("off")
                plt.tight_layout()
                plt.imshow(image[plane][detector])

        if should_save:
            self.save_plot(
                fig=fig,
                event_number=event_number,
                run_number=run_number,
                event_score=event_score,
            )

        if should_show:
            plt.show()

    def save_plot(
        self,
        fig: plt.figure,
        event_number: int,
        run_number: int,
        event_score: float,
        save_path: Optional[str] = None,
    ) -> None:

        save_path = save_path or parameters.final_results_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        event_score = event_score * 100
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", event_score)
        save_path = os.path.join(
            save_path,
            f"{int(event_score)}_{run_number}_{event_number}.png",
        )

        fig.savefig(save_path, dpi=fig.dpi, bbox_inches="tight")
        plt.close(fig)

    def draw_event(
        self,
        event_number: int,
        run_number: int,
        event_score: float,
        should_show: Optional[bool] = None,
        should_save: Optional[bool] = None,
        save_path: Optional[str] = None,
        detectors_side: Optional[str] = None,
        image_dpi: Optional[int] = 300,
        mark_anomalies: Optional[bool] = False,
        is_anomaly: Optional[bool] = True,
        side: Optional[str] = "all",
    ) -> None:

        should_show = should_show if should_show is not None else parameters.should_show
        should_save = should_save if should_save is not None else parameters.should_save
        assert side == "all" or "c" or "a", "In draw_planes side != 'all' or 'a' or 'c'"

        event = self.find_event_in_root(
            event_number=event_number, run_number=run_number
        )

        image_matrix = self.create_image_as_matrix(event)

        side = "a" if np.sum(image_matrix[0][0]) > np.sum(image_matrix[0][3]) else "c"

        fig = plt.figure(dpi=image_dpi)
        fig = self.plot_picture(
            image=image_matrix,
            event_score=event_score,
            event_number=event_number,
            run_number=run_number,
            fig=fig,
            side=side,
            should_show=should_show,
            should_save=should_save,
        )

    def get_extreme_scores_id(
        self,
        max_number: int,
        min_number: int,
        scores_path: str,
        ascending: Optional[bool] = True,
    ) -> Tuple[pd.DataFrame]:

        dask_df = ddf.read_csv(
            scores_path,
            sep=",",
            header=None,
            names=["score", "event_number", "run_number"],
        )

        max_scores = dask_df.nlargest(10, "score").sort_values("score").compute()
        min_scores = dask_df.nsmallest(10, "score").sort_values("score").compute()

        return max_scores, min_scores

    def find_event_in_root(
        self,
        event_number: int,
        run_number: int,
        memory_chunk_size: Optional[str] = parameters.memory_chunk_size,
    ) -> pd.DataFrame:

        root_path = os.path.join(
            parameters.path_to_root_dict, str(run_number) + parameters.root_name_suffix
        )
        with uproot.open(root_path) as file:
            tree = file["TreeHits"]

            for chunk in tree.iterate(
                parameters.preprocess_branches,
                library="pd",
                step_size=memory_chunk_size,
            ):

                return chunk[chunk["evN"] == event_number]

        raise KeyError(
            f"find_event_in_loop(): event_number {event_number} not found in {run_number}"
        )

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

    def draw_examples(self, examples_number: Optional[int] = 10) -> None:

        max_events_id, min_events_id = self.get_extreme_scores_id(
            examples_number, examples_number, parameters.scores_path
        )

        progress = {"current": 0, "total": examples_number * 2}

        for _, event in max_events_id.iterrows():
            self.draw_event(
                event_number=int(event["event_number"]),
                run_number=int(event["run_number"]),
                event_score=event["score"],
                mark_anomalies=True,
                is_anomaly=True,
            )
            if self.should_save:
                progress["current"] = progress["current"] + 1
                print(
                    "Saving anomaly events, progress:",
                    progress["current"],
                    "/",
                    progress["total"],
                )

        for _, event in min_events_id.iterrows():
            self.draw_event(
                event_number=int(event["event_number"]),
                run_number=int(event["run_number"]),
                event_score=event["score"],
                mark_anomalies=True,
                is_anomaly=False,
            )
            if self.should_save:
                progress["current"] = progress["current"] + 1
                print(
                    "Saving normal events, progress:",
                    progress["current"],
                    "/",
                    progress["total"],
                )


def test():

    visualizer = Visualize(
        scores_path=parameters.scores_path,
        anomaly_threshold=0.6,
        should_save=parameters.should_save,
        should_show=parameters.should_show,
        path_to_root_dict=parameters.path_to_root_dict,
        save_path=parameters.final_results_path,
    )

    visualizer.draw_examples(examples_number=5)


if __name__ == "__main__":
    test()
