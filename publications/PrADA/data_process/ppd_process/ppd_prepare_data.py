import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from data_process import data_process_utils
from data_process.ppd_process.ppd_data_creation_config import ppd_data_creation
from data_process.ppd_process.ppd_prepare_data_train_test import prepare_ppd_data


def normalize_df(df):
    column_names = df.columns
    x = df.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def standardize(df_data, df_columns_list, df_cat_mask_list):
    df_list = list()
    df_target = df_data[['target']]
    for columns, is_cat in zip(df_columns_list, df_cat_mask_list):
        df_subset = df_data[columns].copy()
        if is_cat is False:
            df_subset = normalize_df(df_subset)
        df_list.append(df_subset)
    df_list.append(df_target)
    return pd.concat(df_list, axis=1)


def select_positive(data, num_target, select_pos_ratio=0.5):
    target_col = data[:, -1]
    data_1 = data[target_col == 1]
    data_0 = data[target_col == 0]

    print(f"=> select pos samples for target.")
    print(f"| before select positive, data_1: {data_1.shape}")
    print(f"| before select positive, data_0: {data_0.shape}")

    num_positive = int(num_target * select_pos_ratio)
    select_data_1 = data_1[:num_positive]
    test_data_1 = data_1[-400:]
    num_negative = num_target - num_positive
    select_data_0 = data_0[:num_negative]
    test_data_0 = data_0[-3500:]

    print(f"| after select positive, select_data_1: {select_data_1.shape}")
    print(f"| after select positive, select_data_0: {select_data_0.shape}")
    print(f"| after select positive, test_data_1: {test_data_1.shape}")
    print(f"| after select positive, test_data_0: {test_data_0.shape}")

    select_data = np.concatenate((select_data_1, select_data_0), axis=0)
    left_data_for_test = np.concatenate((test_data_1, test_data_0), axis=0)
    return select_data, left_data_for_test


def combine_src_tgt_data(from_dir, to_dir, data_tag):
    print(f"========================= combine ppd source and target data ============================ ")

    source_train_file_name = from_dir + f"PPD_2014_src_1to9_ad_{data_tag}_train.csv"
    target_train_file_name = from_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_train.csv'

    df_src_data = pd.read_csv(source_train_file_name, skipinitialspace=True)
    df_tgt_data = pd.read_csv(target_train_file_name, skipinitialspace=True)
    print("[INFO] df_src_data shape:", df_src_data.shape)
    print("[INFO] df_tgt_data shape:", df_tgt_data.shape)

    df_data = data_process_utils.combine_src_tgt_data(df_src_data, df_tgt_data)
    print("[INFO] df_src_tgt_data shape:", df_data.shape)

    file_full_name = "{}/PPD_2014_src_tgt_{}_{}.csv".format(to_dir, data_tag, "train")
    data_process_utils.save_df_data(df_data, file_full_name)


def create_ppd_src_tgt_data(df_dict,
                            df_column_split_list,
                            df_cat_mask_list,
                            to_dir,
                            data_config,
                            all_col_list,
                            train=True,
                            data_2014_tgt_10to12_test=None):
    data_mode = "train" if train else "valid"
    print(f"========================= create_ppd_source_target_data for {data_mode} data ======================== ")

    num_tgt = data_config['num_tgt']
    select_tgt_pos_ratio = data_config['select_tgt_pos_ratio']
    data_tag = data_config['data_tag']

    df_2014_1to9 = df_dict['df_2014_1to9']
    df_2014_10to12 = df_dict['df_2014_10to12']

    print("[INFO] (original) source: df_2014_1to9:", df_2014_1to9.shape)
    print("[INFO] (original) target: df_2014_10to12:", df_2014_10to12.shape)

    data_2014_src_1to9_ad = df_2014_1to9.values
    data_2014_tgt_10to12 = df_2014_10to12.values

    print("[INFO] select positive samples for target data.")
    data_2014_tgt_10to12_for_test = None
    if train:
        data_2014_tgt_10to12_ft, data_2014_tgt_10to12_for_test = select_positive(data_2014_tgt_10to12,
                                                                                 num_target=num_tgt,
                                                                                 select_pos_ratio=select_tgt_pos_ratio)
    else:
        # valid
        data_2014_tgt_10to12_ft = np.concatenate((data_2014_tgt_10to12, data_2014_tgt_10to12_test), axis=0)

    # shuffle
    data_2014_src_1to9_ad = shuffle(data_2014_src_1to9_ad)
    target_col = data_2014_src_1to9_ad[:, -1]
    data_1 = data_2014_src_1to9_ad[target_col == 1]
    data_0 = data_2014_src_1to9_ad[target_col == 0]
    data_2014_src_1to9_ad = shuffle(np.concatenate((data_1, data_0[:37000]), axis=0))

    target_col = data_2014_src_1to9_ad[:, -1]
    data_1 = data_2014_src_1to9_ad[target_col == 1]
    data_0 = data_2014_src_1to9_ad[target_col == 0]
    print("[INFO] data_2014_src_1to9_ad:", data_2014_src_1to9_ad.shape)
    print("[INFO] data_2014_src_1to9_ad pos:", data_1.shape)
    print("[INFO] data_2014_src_1to9_ad neg:", data_0.shape)

    data_2014_tgt_10to12_ad = data_2014_tgt_10to12
    target_col = data_2014_tgt_10to12_ad[:, -1]
    data_1 = data_2014_tgt_10to12_ad[target_col == 1]
    data_0 = data_2014_tgt_10to12_ad[target_col == 0]
    data_2014_tgt_10to12_ad = shuffle(np.concatenate((data_1, data_0[:8000]), axis=0))

    target_col = data_2014_tgt_10to12_ad[:, -1]
    data_1 = data_2014_tgt_10to12_ad[target_col == 1]
    data_0 = data_2014_tgt_10to12_ad[target_col == 0]
    print("[INFO] data_2014_tgt_10to12_ad:", data_2014_tgt_10to12_ad.shape)
    print("[INFO] data_2014_tgt_10to12_ad pos:", data_1.shape)
    print("[INFO] data_2014_tgt_10to12_ad neg:", data_0.shape)

    data_2014_tgt_10to12_ft = shuffle(data_2014_tgt_10to12_ft)
    target_col = data_2014_tgt_10to12_ft[:, -1]
    data_1 = data_2014_tgt_10to12_ft[target_col == 1]
    data_0 = data_2014_tgt_10to12_ft[target_col == 0]
    print("[INFO] data_2014_tgt_10to12_ft:", data_2014_tgt_10to12_ft.shape)
    print("[INFO] data_2014_tgt_10to12_ft pos:", data_1.shape)
    print("[INFO] data_2014_tgt_10to12_ft neg:", data_0.shape)

    # for source domain adaptation and source classification
    df_2014_src_1to9_ad = pd.DataFrame(data=data_2014_src_1to9_ad, columns=all_col_list)

    # target domain adaptation
    df_2014_tgt_10to12_ad = pd.DataFrame(data=data_2014_tgt_10to12_ad, columns=all_col_list)

    # target classification (fine-tune)
    df_2014_tgt_10to12_ft = pd.DataFrame(data=data_2014_tgt_10to12_ft, columns=all_col_list)

    # standardize
    df_2014_src_1to9_ad = standardize(df_2014_src_1to9_ad, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_10to12_ad = standardize(df_2014_tgt_10to12_ad, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_10to12_ft = standardize(df_2014_tgt_10to12_ft, df_column_split_list, df_cat_mask_list)

    print("[INFO] (final) df_2014_src_1to9_ad: ", df_2014_src_1to9_ad.shape)
    print("[INFO] (final) df_2014_tgt_10to12_ad: ", df_2014_tgt_10to12_ad.shape)

    # save
    file_full_name = "{}/PPD_2014_src_1to9_ad_{}_{}.csv".format(to_dir, data_tag, data_mode)
    df_2014_src_1to9_ad.to_csv(file_full_name, index=False)
    print(f"[INFO] save df_2014_src_1to9_ad to {file_full_name}")

    file_full_name = "{}/PPD_2014_tgt_10to12_ad_{}_{}.csv".format(to_dir, data_tag, data_mode)
    df_2014_tgt_10to12_ad.to_csv(file_full_name, index=False)
    print(f"[INFO] save data_2014_tgt_10to12_ad to {file_full_name}")

    if train:
        print("[INFO] (final) df_2014_tgt_10to12_ft: ", df_2014_tgt_10to12_ft.shape)
        file_full_name = "{}/PPD_2014_tgt_10to12_ft_{}_{}.csv".format(to_dir, data_tag, data_mode)
        df_2014_tgt_10to12_ft.to_csv(file_full_name, index=False)
        print(f"[INFO] save data_2014_tgt_10to12_ft to {file_full_name}")
    else:
        half_num = int(df_2014_tgt_10to12_ft.shape[0] / 2)
        df_2014_tgt_10to12_ft_valid = df_2014_tgt_10to12_ft[:half_num]
        df_2014_tgt_10to12_ft_test = df_2014_tgt_10to12_ft[half_num:]

        print("[INFO] (final) df_2014_tgt_10to12_ft_valid: ", df_2014_tgt_10to12_ft_valid.shape)
        print("[INFO] (final) df_2014_tgt_10to12_ft_test: ", df_2014_tgt_10to12_ft_test.shape)

        file_full_name = "{}/PPD_2014_tgt_10to12_ft_{}_{}.csv".format(to_dir, data_tag, "valid")
        df_2014_tgt_10to12_ft_valid.to_csv(file_full_name, index=False)
        print(f"[INFO] save df_2014_tgt_10to12_ft_valid to {file_full_name}")

        file_full_name = "{}/PPD_2014_tgt_10to12_ft_{}_{}.csv".format(to_dir, data_tag, "test")
        df_2014_tgt_10to12_ft_test.to_csv(file_full_name, index=False)
        print(f"[INFO] save df_2014_tgt_10to12_ft_test to {file_full_name}")

    return data_2014_tgt_10to12_for_test


if __name__ == "__main__":
    print(f"[INFO] ppd_data_creation config:{ppd_data_creation}")

    # prepare_ppd_data()
    # from_dir = ppd_data_creation['processed_data_dir']

    from_dir = ppd_data_creation['original_data_dir']
    output_dir = ppd_data_creation['processed_data_dir']
    meta_data_full_name = ppd_data_creation['meta_data_full_name']
    num_train = ppd_data_creation['number_train_samples']
    num_tgt = ppd_data_creation['number_target_samples']
    select_tgt_pos_ratio = ppd_data_creation['positive_samples_ratio']
    data_tag = ppd_data_creation['data_tag']

    with open(meta_data_full_name) as json_file:
        print(f"[INFO] load task meta file from {meta_data_full_name}")
        meta_data_dict = json.load(json_file)
    print("[INFO] meta_data_dict", meta_data_dict)

    df_cat_mask_list = meta_data_dict['df_cat_mask_list']
    df_column_split_list = meta_data_dict['df_column_split_list']
    # df_all_column_list = meta_data_dict['df_all_column_list'] + ['target']

    df_2014_train = pd.read_csv(from_dir + f"PPD_data_2014_{num_train}_train.csv", skipinitialspace=True)
    df_2014_test = pd.read_csv(from_dir + f"PPD_data_2014_{num_train}_test.csv", skipinitialspace=True)
    df_datetime_2014_train = pd.read_csv(from_dir + f"PPD_datetime_2014_{num_train}_train.csv", skipinitialspace=True)
    df_datetime_2014_test = pd.read_csv(from_dir + f"PPD_datetime_2014_{num_train}_test.csv", skipinitialspace=True)

    print(f"[INFO] df_2014_train.shape:{df_2014_train.shape}")
    print(f"[INFO] df_2014_test.shape:{df_2014_test.shape}")
    print(f"[INFO] df_datetime_2014_train.shape:{df_datetime_2014_train.shape}")
    print(f"[INFO] df_datetime_2014_test.shape:{df_datetime_2014_test.shape}")

    df_2014_1to9_train = df_2014_train[df_datetime_2014_train['ListingInfo_Month'] <= 9]
    df_2014_10to12_train = df_2014_train[df_datetime_2014_train['ListingInfo_Month'] >= 10]
    df_2014_1to9_test = df_2014_test[df_datetime_2014_test['ListingInfo_Month'] <= 9]
    df_2014_10to12_test = df_2014_test[df_datetime_2014_test['ListingInfo_Month'] >= 10]

    df_train_dict = {
        "df_2014_1to9": df_2014_1to9_train,
        "df_2014_10to12": df_2014_10to12_train
    }

    df_test_dict = {
        "df_2014_1to9": df_2014_1to9_test,
        "df_2014_10to12": df_2014_10to12_test
    }

    data_config = {"select_tgt_pos_ratio": select_tgt_pos_ratio,
                   "num_tgt": num_tgt,
                   "data_tag": data_tag}

    data_2014_tgt_10to12_for_test = create_ppd_src_tgt_data(df_train_dict,
                                                            df_column_split_list,
                                                            df_cat_mask_list,
                                                            output_dir,
                                                            data_config,
                                                            all_col_list=df_2014_train.columns,
                                                            train=True)
    create_ppd_src_tgt_data(df_test_dict,
                            df_column_split_list,
                            df_cat_mask_list,
                            output_dir,
                            data_config,
                            all_col_list=df_2014_train.columns,
                            train=False,
                            data_2014_tgt_10to12_test=data_2014_tgt_10to12_for_test)

    # NOTE: input dir and output dir are the same for src_tgt_data
    combine_src_tgt_data(from_dir=output_dir, to_dir=output_dir, data_tag=data_tag)
