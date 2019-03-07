from utils import *


class Aggregation():
    def __init__(self, categorical_column, num_aggregations):
        self.categorical_column = categorical_column

        self.num_aggregations = num_aggregations
        self.numeric_columns = list(self.num_aggregations.keys())

    def create_features(self, df):
        gp = df[[self.categorical_column] + self.numeric_columns].groupby([self.categorical_column])
        agg = gp.agg({**self.num_aggregations})
        new_cols = ['agg_{}_{}_{}'.format(self.categorical_column, e[0], e[1].upper()) for e in agg.columns.tolist()]
        agg.columns = pd.Index(new_cols)
        df = df.merge(agg, how="left", on=self.categorical_column)

        return df[['PetID']+new_cols]


if __name__ == "__main__":
    aggs = {
        'MaturitySize': ['min', 'max', 'mean', "sum"],
        'FurLength': ['min', 'max', 'mean', "sum"],
    #    'Vaccinated': ['min', 'max', 'mean', "sum"],
    #    'Dewormed': ['min', 'max', 'mean', "sum"],
    #    'Sterilized': ['min', 'max', 'mean', "sum"],
        'Health': ['min', 'max', 'mean', "sum"],
    #    'Quantity': ['min', 'max', 'mean', "sum"],
        'Fee': ['min', 'max', 'mean', "sum"],
    }

    categorical_column = 'State'
    aggregator = Aggregation(categorical_column, aggs)
    all_data_agg = aggregator.create_features(train)
    all_data_agg.to_feather("../feature/state_agg.feather")