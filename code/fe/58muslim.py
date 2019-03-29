import pandas as pd
from utils import *

df = pd.read_csv("../feature/state_labels_with_muslim.csv")
cols = df.columns.drop(["StateID", "StateName"])
for c in cols:
    df[c] = df[c].str.replace(",", "").astype("float32")

train.merge(df, how="left", left_on="State", right_on="StateID")[cols].to_feather("../feature/muslim.feather")