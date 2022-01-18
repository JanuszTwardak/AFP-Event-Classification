import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest

MAX_SAMPLES = 3

INPUT_PATH = "preprocessed_data/341294_afp_hits.pkl"

# Load the two datasets
df = pd.read_pickle(INPUT_PATH)
df = df.iloc[:6]
df.head()

rng = np.random.RandomState(0)

# Helper function to train and predict IF model for a feature
def train_and_predict_if(df, feature):
    clf = IsolationForest(max_samples=MAX_SAMPLES, random_state=rng)
    clf.fit(df[[feature]])
    pred = clf.predict(df[[feature]])
    scores = clf.decision_function(df[[feature]])
    stats = pd.DataFrame()
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


# Run through all features and save the outlier scores for each feature
num_columns = [
    i
    for i in list(df.columns)
    if i not in list(df.select_dtypes("object").columns) and i not in ["Id"]
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
n_outliers = 1
print_outliers(outliers, "hits_n", n_outliers)
print_outliers(outliers, "hit_row_1", n_outliers)
print_outliers(outliers, "hit_row_2", n_outliers)
print_outliers(outliers, "hit_column_1", n_outliers)
