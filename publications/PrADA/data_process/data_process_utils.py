import pandas as pd
from sklearn.utils import shuffle


def save_df_data(df_data, file_full_name):
    df_data.to_csv(file_full_name, index=False)
    print(f"[INFO] save data with shape {df_data.shape} to {file_full_name}")


def combine_src_tgt_data(df_src_data, df_tgt_data):
    df_all_data = pd.concat((df_src_data, df_tgt_data), axis=0)
    df_all_data = shuffle(df_all_data)
    return df_all_data
