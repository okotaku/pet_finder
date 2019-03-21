from utils import *
import feather

all_data = feather.read_dataframe("../from_kernel/all_data.feather")
t_cols = np.load("../from_kernel/t_cols.npy")

mean_cols = [c for c in t_cols if re.match("mean_\w*groupby", c)]
median_cols = [c.replace("mean", "median") for c in mean_cols]
min_cols = [c.replace("mean", "min") for c in mean_cols]
max_cols = [c.replace("mean", "max") for c in mean_cols]

max_min_cols = [c.replace("mean", "max_diff_min") for c in mean_cols]
max_mean_cols = [c.replace("mean", "max_diff_mean") for c in mean_cols]
mean_min_cols = [c.replace("mean", "mean_diff_min") for c in mean_cols]

df_ = pd.DataFrame(all_data[max_cols].values - all_data[min_cols].values, columns=max_min_cols)
df_.to_feather("../feature/max_diff_min.feather")

df_ = pd.DataFrame(all_data[max_cols].values - all_data[mean_cols].values, columns=max_mean_cols)
df_.to_feather("../feature/max_diff_mean.feather")

df_ = pd.DataFrame(all_data[mean_cols].values - all_data[min_cols].values, columns=mean_min_cols)
df_.to_feather("../feature/mean_diff_min.feather")