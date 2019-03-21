from utils import *

breeds = pd.read_csv('../../input/breed-labels-with-ranks/breed_labels_with_ranks.csv')
train = train.merge(breeds, how="left", left_on="fix_Breed1", right_on="BreedID")
train = train.rename(columns={"BreedCatRank": "BreedCatRank_main", "BreedDogRank": "BreedDogRank_main"})
train = train.merge(breeds, how="left", left_on="fix_Breed2", right_on="BreedID")
train = train.rename(columns={"BreedCatRank": "BreedCatRank_second", "BreedDogRank": "BreedDogRank_second"})

train[["BreedCatRank_main", "BreedDogRank_main", "BreedCatRank_second", "BreedDogRank_second"]].to_feather("../feature/breedrank.feather")