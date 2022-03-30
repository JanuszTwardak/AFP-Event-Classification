import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest

import uproot
from matplotlib import pyplot as plt
import sys

sys.path.append(".")
from Visualize import Visualize


rng = np.random.RandomState(0)

# Helper function to train and predict IF model for a feature
def train_and_predict_if(df, feature):
    """_summary_

    :param df: _description_
    :type df: _type_
    :param feature: _description_
    :type feature: _type_
    :return: _description_
    :rtype: _type_
    """
    clf = IsolationForest(random_state=rng)
    print(df[[feature]])
    clf.fit(df[[feature]])
    pred = clf.predict(df[[feature]])
    scores = clf.decision_function(df[[feature]])
    stats = pd.DataFrame()
    stats["evN"] = df["evN"]
    stats["side"] = df["side"]
    stats["val"] = df[feature]
    stats["score"] = scores
    stats["outlier"] = pred
    stats["min"] = df[feature].min()
    stats["max"] = df[feature].max()
    stats["mean"] = df[feature].mean()
    stats["feature"] = [feature] * len(df)
    return stats


# Helper function to print outliers
def print_outliers(df, feature, n):
    print(feature)
    print(df[feature].head(n).to_string(), "\n")


def print_all_outliers(df, feature):
    print(feature)
    print(df[feature].to_string(), "\n")


# Run through all features and save the outlier scores for each feature


# event_number = 1344083997
# print(full_dataset.loc[full_dataset["evN"] == event_number])
# print(df.loc[df["evN"] == event_number])


def main():
    ROOT_FILE_NAME = "336505_afp_newhits"
    PREPROCESSED_INPUT_PATH = "preprocessed_data/" + ROOT_FILE_NAME + ".pkl"

    # Load the two datasets
    df = pd.read_pickle(PREPROCESSED_INPUT_PATH)
    print(df.head())

    df = df.dropna()
    num_columns = [
        i
        for i in list(df.columns)
        if i not in list(df.select_dtypes("object").columns)
        and i not in ["evN", "side"]
    ]

    result = pd.DataFrame()

    for feature in num_columns:
        stats = train_and_predict_if(df, feature)
        result = pd.concat([result, stats])

    # Gather top outliers for each feature
    outliers = {
        team: grp.drop("feature", axis=1)
        for team, grp in result.sort_values(by="score").groupby("feature")
    }

    # Print the top 10 outlier samples for a few selected features
    n_outliers = 10
    print_outliers(outliers, "hits_n", n_outliers)
    print_outliers(outliers, "hit_row_1", n_outliers)
    print_outliers(outliers, "hit_row_2", n_outliers)
    print_outliers(outliers, "hit_column_1", n_outliers)


if __name__ == "__main__":
    main()
    visualize = Visualize(
        input_dataset="336505_afp_newhits", should_save=True, should_show=True
    )

    i = 0

    visualize.draw_planes(853927973)
