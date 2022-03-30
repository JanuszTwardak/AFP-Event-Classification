from typing import Optional, Union
import numpy as np
from matplotlib import pyplot as plt
import uproot
import time
import pandas as pd
from pathlib import Path


class Visualize:
    def __init__(
        self,
        input_dataset: Union[pd.DataFrame, str],
        should_save: bool,
        should_show: bool,
        scores: Union[pd.DataFrame, str],
        anomaly_threshold: float,
        root_name: str,  # TODO change to optional parameter in final version and add such handling
    ) -> None:

        self.should_save = should_save
        self.should_show = should_show
        self.dataset = self.load_dataset(input_dataset)
        self.scores = self.load_scores(scores)
        self.root_name = root_name
        self.anomaly_threshold = anomaly_threshold

    def load_dataset(self, input_dataset: Union[pd.DataFrame, str]) -> pd.DataFrame:

        return (
            input_dataset
            if isinstance(input_dataset, pd.DataFrame)
            else self.load_root_from_path(input_dataset)
        )

    @staticmethod
    def load_scores(input_scores: Union[pd.DataFrame, str]) -> pd.DataFrame:

        return (
            input_scores
            if isinstance(input_scores, pd.DataFrame)
            else pd.read_pickle(input_scores)
        )

    @staticmethod
    def load_root_from_path(root_name: str) -> pd.DataFrame:
        ROOT_INPUT_PATH = "input_root/" + root_name + ".root"

        file = uproot.open(ROOT_INPUT_PATH)
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

                hit_rows.set_axis(["value"], axis=1, inplace=True)
                hit_columns.set_axis(["value"], axis=1, inplace=True)

                for index, y in enumerate(hit_rows["value"]):
                    if y != -1:
                        x = hit_columns["value"].iloc[index]
                        # TODO: charge_value instead of fixed 1 it should hold normalised value of charge
                        charge_value = 1
                        image_matrix[detector, plane, y, x] = charge_value

        return image_matrix

    def draw_planes(
        self,
        event_number: int,
        should_show: Optional[bool] = None,
        should_save: Optional[bool] = None,
        save_path: Optional[str] = None,
        detectors_side: Optional[str] = None,
        image_dpi: Optional[int] = 300,
        mark_anomalies: Optional[bool] = True,
    ) -> None:

        should_show = should_show or self.should_show
        should_save = should_save or self.should_save

        event = self.dataset.loc[self.dataset["evN"] == event_number]
        # event["a_hits_n"] = event.filter(regex="^hits\\[[01]", axis=1).sum(axis=1)
        # event["c_hits_n"] = event.filter(regex="^hits\\[[23]", axis=1).sum(axis=1)

        image_matrix = self.create_image_as_matrix(event)

        fig = plt.figure(dpi=image_dpi)
        rows = 2
        columns = 8

        position = 1
        for detector in range(4):
            for plane in range(4):
                fig.add_subplot(rows, columns, position)
                position += 1
                # TODO plane and detector should be swapped, probably some error?
                plt.imshow(image_matrix[plane][detector])
                plt.axis("off")
                plt.tight_layout()

        if self.should_save:
            save_path = save_path or ("/output/" + self.root_name)
            save_path = save_path + "/" if save_path[-1] != "/" else save_path
            Path(save_path).mkdir(parents=True, exist_ok=True)

            if mark_anomalies:
                is_anomaly = "anomaly" if is_anomaly else "normal"
            else:
                is_anomaly = ""
            timestr = time.strftime("%Y%m%d-%H%M%S")
            save_path = (
                save_path + str(event_number) + "_" + timestr + is_anomaly + ".png"
            )
            plt.savefig(save_path)

        if self.should_show:
            plt.show()

        # TODO: draw_examples([...])
        # def draw_examples(
        #     dataset: pd.DataFrame, scores: pd.DataFrame, examples_number: Optional[int] = 10
        # ):
        #     scores.sort_values(by="score", ascending=True)

        #     for image_counter in range(examples_number):
