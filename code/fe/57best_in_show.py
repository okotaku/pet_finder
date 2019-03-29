import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import *

df = pd.read_csv("../feature/best_in_show.csv")
df["food_costs_per_year"] = df["food_costs_per_year"].str.replace("$", "").str.replace(",", "").fillna(-999).astype("int32").replace(-999, np.nan)
df[list(df.columns.drop(["Dog breed", "category", "size_category"]))] = df[list(df.columns.drop(["Dog breed", "category", "size_category"]))].astype("float32")
for c in ["category", "size_category"]:
    df[c] = LabelEncoder().fit_transform(df[c].fillna("nan").values)
breed = pd.read_csv("../../input/petfinder-adoption-prediction/breed_labels.csv")

new_cols = list(df.columns.drop("Dog breed"))
breed = breed.merge(df, how="left", left_on="BreedName", right_on="Dog breed")

orig_cols = train.columns
train = train.merge(breed[["BreedID"]+new_cols], how="left", left_on="fix_Breed1", right_on="BreedID").drop("BreedID", axis=1)
dic = {}
for c in new_cols:
    dic[c] = c + "_main"
train = train.rename(columns=dic)

train = train.merge(breed[["BreedID"]+new_cols], how="left", left_on="fix_Breed2", right_on="BreedID")
train = train.rename(columns={"BreedCatRank": "BreedCatRank_second", "BreedDogRank": "BreedDogRank_second"}).drop("BreedID", axis=1)
for c in new_cols:
    dic[c] = c + "_second"
train = train.rename(columns=dic)

new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].to_feather("../feature/best_in_show.feather")