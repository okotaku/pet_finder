import feather
from utils import *

df = feather.read_dataframe("../feature/dense121_2_X.feather")
agg = df.groupby("PetID").agg(["min", "max", "var", "sum"])
new_cols = ["{}_{}".format(c[0], c[1]) for c in agg.columns]
agg.columns = new_cols
agg = agg.reset_index()

train = train.merge(agg, how="left", on="PetID")
train[new_cols].to_feather("../feature/denseagg.feather")