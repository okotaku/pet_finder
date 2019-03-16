from utils import *


def merge_state_info(labels_state, all_data):
    labels_state.rename(columns={'StateID': 'State'}, inplace=True)
    all_data_state = pd.merge(all_data, labels_state, on='State', how="left")

    state_info = pd.read_csv('../../input/state_info/state_info.csv')
    state_info.rename(columns={
        "Area (km2)": "Area",
        "Pop. density": "Pop_density",
        "Urban pop.(%)": "Urban_pop",
        "Bumiputra (%)": "Bumiputra",
        "Chinese (%)": "Chinese",
        "Indian (%)": "Indian"
    }, inplace=True)
    for key in ["Population", "Area", "Pop_density", "2017GDPperCapita"]:
        state_info[key] = state_info[key].fillna("-999").str.replace(",", "").astype("int32").replace(-999, np.nan)
    state_info["StateName"] = state_info["StateName"].str.replace("FT ", "")
    state_info["StateName"] = state_info["StateName"].str.replace("Malacca", "Melaka")
    state_info["StateName"] = state_info["StateName"].str.replace("Penang", "Pulau Pinang")
    new_cols = list(state_info.columns)
    new_cols.remove("StateName")
    all_data_state = pd.merge(all_data_state, state_info, on='StateName', how="left")

    return all_data_state[['PetID'] + new_cols]


if __name__ == "__main__":
    all_data_state = merge_state_info(labels_state, train)
    all_data_state.to_feather("../feature/state_info.feather")