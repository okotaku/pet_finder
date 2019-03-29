from utils import *
from sklearn.preprocessing import LabelEncoder

char_cat = pd.read_csv("../feature/cat_breed_characteristics.csv")
char_dog = pd.read_csv("../feature/dog_breed_characteristics.csv")
char = char_cat.append(char_dog).reset_index(drop=True)
cat = ["Fur", "Group1", "Group2", "LapCat"]
adv_cat = ["BreedName", "Temperment", "AltBreedName"]
char[list(char.columns.drop(cat+adv_cat))] = char[list(char.columns.drop(cat+adv_cat))].astype("float32")
for c in cat:
    char[c] = LabelEncoder().fit_transform(char[c].fillna("nan").values)
breed = pd.read_csv("../../input/petfinder-adoption-prediction/breed_labels.csv")

breed = breed.merge(char, how="left", left_on="BreedName", right_on="BreedName")
new_cols = list(char.columns.drop(["BreedName", "AltBreedName"]))

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
train[new_cols].to_feather("../feature/char.feather")