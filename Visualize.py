import numpy as np
from matplotlib import pyplot as plt
import uproot


class Visualize:
    def __init__(self, root_file_name, should_save, should_show):
        self.should_save = should_save
        self.should_show = should_show
        self.root_file_name = root_file_name

        ENTRY_LIMIT = 100  # TODO DELETE AFTER TESTING
        ROOT_INPUT_PATH = "input_root/" + self.root_file_name + ".root"

        file = uproot.open(ROOT_INPUT_PATH)
        tree = file["TreeHits"]
        raw_dataset = tree.arrays(
            ["evN", "hits", "hits_row", "hits_col", "hits_q"],
            library="pd",
            entry_stop=ENTRY_LIMIT,  # TODO DELETE AFTER TESTING
        )

        self.raw_dataset = raw_dataset

    def load_root(self):
        ENTRY_LIMIT = 100  # TODO DELETE AFTER TESTING
        ROOT_INPUT_PATH = "input_root/" + self.root_file_name + ".root"

        file = uproot.open(ROOT_INPUT_PATH)
        tree = file["TreeHits"]
        raw_dataset = tree.arrays(
            ["evN", "hits", "hits_row", "hits_col", "hits_q"],
            library="pd",
            entry_stop=ENTRY_LIMIT,  # TODO DELETE AFTER TESTING
        )
        return raw_dataset

    def draw_planes(self, event_number):
        event_matrix = np.zeros(shape=(4, 4, 336, 80))

        event = self.raw_dataset.loc[self.raw_dataset["evN"] == event_number]
        # event["a_hits_n"] = event.filter(regex="^hits\\[[01]", axis=1).sum(axis=1)
        # event["c_hits_n"] = event.filter(regex="^hits\\[[23]", axis=1).sum(axis=1)

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

                hit_rows.rename(columns={4: "value"}, inplace=True)
                hit_columns.rename(columns={4: "value"}, inplace=True)

                for index, y in enumerate(hit_rows["value"]):
                    if y != -1:
                        x = hit_columns["value"].iloc[index]
                        charge_value = 1
                        event_matrix[detector, plane, y, x] = charge_value

        fig = plt.figure(dpi=300)
        rows = 2
        columns = 8

        position = 1
        for detector in range(4):
            for plane in range(4):
                fig.add_subplot(rows, columns, position)
                position += 1
                # TODO plane and detector should be swapped, probably saving error?
                plt.imshow(event_matrix[plane][detector])
                plt.axis("off")
                plt.tight_layout()

        if self.should_save:
            name = "output/" + self.root_file_name + "_" + str(event_number) + ".png"
            print(name)
            plt.savefig(name)

        if self.should_show:
            plt.show()
