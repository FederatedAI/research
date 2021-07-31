import numpy as np
import pandas as pd
from sklearn import preprocessing

from data_process.census_process.mapping_resource import education_value_map, workclass_map, education_map, \
    marital_status_map, occupation_map, race_map, country_map, \
    census_income_label
from data_process.census_process.utils import bucketized_age


def consistentize_census9495_columns(data_frame):
    # data_frame = data_frame.replace(to_replace="Not in universe", value="None")
    data_frame = data_frame.replace(to_replace="?", value="None")
    data_frame = data_frame.replace(to_replace=np.NaN, value="None")
    data_frame = data_frame.replace({"class_worker": workclass_map, "education": education_map,
                                     "marital_stat": marital_status_map, "major_occ_code": occupation_map,
                                     "country_father": country_map, "country_mother": country_map,
                                     "country_self": country_map, "race": race_map,
                                     "income_label": census_income_label})
    return data_frame


def numericalize_census9495_data(data_frame, to_index_map):
    data_frame['education_year'] = data_frame.apply(lambda row: education_value_map[row.education], axis=1)
    data_frame['age_index'] = data_frame.apply(lambda row: bucketized_age(row.age), axis=1)

    # convert categorical to index
    data_frame = data_frame.replace(to_index_map)
    data_frame['gender_index'] = data_frame['gender']

    return data_frame


global_scaler = None


def standardize_census_data(data_frame, cols_to_standardize, train_scaler=None):
    feat = data_frame[cols_to_standardize].values
    scaler = preprocessing.StandardScaler() if train_scaler is None else train_scaler
    s_feat = scaler.fit_transform(feat)  # axis=0 for column-wise
    for idx in range(len(cols_to_standardize)):
        data_frame[cols_to_standardize[idx]] = s_feat[:, idx]
    return data_frame, scaler


def split_data_based_on_year(data):
    data94 = data[data["year"] == 94]
    data95 = data[data["year"] == 95]
    return data94, data95


def sample_data(data, num_samples=32561):
    columns = data.columns

    pos_df = data[data["income_label"] == 1.0]
    neg_df = data[data["income_label"] == 0.0]

    num_neg_data_needed = num_samples - len(pos_df)
    pos_data = pos_df.values
    neg_data = neg_df.values

    print("num_neg_data_needed:", num_neg_data_needed)
    idxs = [i for i in range(len(neg_data))]
    sampled_idxs = np.random.choice(idxs, num_neg_data_needed, replace=False)
    sampled_neg_Data = neg_data[sampled_idxs]

    sampled_data = np.concatenate([pos_data, sampled_neg_Data], axis=0)

    sampled_df = pd.DataFrame(sampled_data, columns=columns)
    return sampled_df


def sample_data_v2(data, num_samples=32561):
    pos_df = data[data["income_label"] == 1.0]
    neg_df = data[data["income_label"] == 0.0]
    print(f"data shape:{data.shape} with pos:{pos_df.shape} and neg:{neg_df.shape}")

    columns = data.columns
    data = data.values
    idxs = [i for i in range(len(data))]
    sampled_idxs = np.random.choice(idxs, num_samples, replace=False)
    sampled_data = data[sampled_idxs]

    sampled_df = pd.DataFrame(sampled_data, columns=columns)

    pos_df = sampled_df[sampled_df["income_label"] == 1.0]
    neg_df = sampled_df[sampled_df["income_label"] == 0.0]
    print(f"data shape:{data.shape} with pos:{pos_df.shape} and neg:{neg_df.shape}")

    return sampled_df


def sample_data_v3(data, pos_num_samples, neg_num_samples):
    pos_df = data[data["income_label"] == 1.0]
    neg_df = data[data["income_label"] == 0.0]

    print(f"data shape:{data.shape} with pos:{pos_df.shape} and neg:{neg_df.shape}")
    print(f"select {pos_num_samples} pos samples from pos:{pos_df.shape}")
    print(f"select {neg_num_samples} neg samples from neg:{neg_df.shape}")

    columns = data.columns
    # data = data.values
    pos_data = pos_df.values
    neg_data = neg_df.values

    pos_idxs = [i for i in range(len(pos_data))]
    neg_idxs = [i for i in range(len(neg_data))]

    sampled_pos_idxs = np.random.choice(pos_idxs, pos_num_samples, replace=False)
    sampled_neg_idxs = np.random.choice(neg_idxs, neg_num_samples, replace=False)

    sampled_pos_data = pos_data[sampled_pos_idxs]
    sampled_neg_data = neg_data[sampled_neg_idxs]

    sampled_data = np.concatenate([sampled_pos_data, sampled_neg_data], axis=0)
    np.random.shuffle(sampled_data)

    sampled_df = pd.DataFrame(sampled_data, columns=columns)

    pos_df = sampled_df[sampled_df["income_label"] == 1.0]
    neg_df = sampled_df[sampled_df["income_label"] == 0.0]
    print(f"data shape:{data.shape} with pos:{pos_df.shape} and neg:{neg_df.shape}")

    return sampled_df


if __name__ == "__main__":
    print(bucketized_age(4))
    print(bucketized_age(20))
    print(bucketized_age(28))
    print(bucketized_age(33))
    print(bucketized_age(38))
    print(bucketized_age(41))
    print(bucketized_age(46))
    print(bucketized_age(53))
    print(bucketized_age(58))
    print(bucketized_age(61))
    print(bucketized_age(66))
