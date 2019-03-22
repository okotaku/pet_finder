import pandas as pd
from utils import *

df = pd.read_csv("../feature/Lifespan.csv")
breed = pd.read_csv("../../input/petfinder-adoption-prediction/breed_labels.csv")

orig_cols = train.columns
train = train.merge(df[["BreedID", "Lifespan", "Senior"]], how="left", left_on="fix_Breed1", right_on="BreedID").drop("BreedID", axis=1)
train = train.rename(columns={"Lifespan": "Lifespan_main", "Senior": "Senior_main"})
train = train.merge(df[["BreedID", "Lifespan", "Senior"]], how="left", left_on="fix_Breed2", right_on="BreedID")
train = train.rename(columns={"BreedCatRank": "BreedCatRank_second", "BreedDogRank": "BreedDogRank_second"}).drop("BreedID", axis=1)
new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].to_feather("../feature/lifespan.feather")