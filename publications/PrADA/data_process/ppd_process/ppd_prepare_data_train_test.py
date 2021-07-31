import pandas as pd
from sklearn.utils import shuffle

from data_process.ppd_process.ppd_data_creation_config import ppd_data_creation


def create_train_and_test(df_data, df_datetime, num_train, to_dir):

    year = 2014
    df_data_2014 = df_data[df_datetime['ListingInfo_Year'] == year]
    df_datetime_2014 = df_datetime[df_datetime['ListingInfo_Year'] == year]

    df_data_2014, df_datetime_2014 = shuffle(df_data_2014, df_datetime_2014)

    df_data_train = df_data_2014[:num_train]
    df_datetime_train = df_datetime_2014[:num_train]

    df_data_test = df_data_2014[num_train:]
    df_datetime_test = df_datetime_2014[num_train:]

    print(f"[INFO] df_data_train with shape: {df_data_train.shape}")
    print(f"[INFO] df_data_test with shape: {df_data_test.shape}")
    print(f"[INFO] df_datetime_train with shape: {df_datetime_train.shape}")
    print(f"[INFO] df_datetime_test with shape: {df_datetime_test.shape}")

    tag = str(year)
    df_data_train.to_csv("{}/PPD_data_{}_{}_train.csv".format(to_dir, tag, str(num_train)), index=False)
    df_data_test.to_csv("{}/PPD_data_{}_{}_test.csv".format(to_dir, tag, str(num_train)), index=False)
    df_datetime_train.to_csv("{}/PPD_datetime_{}_{}_train.csv".format(to_dir, tag, str(num_train)), index=False)
    df_datetime_test.to_csv("{}/PPD_datetime_{}_{}_test.csv".format(to_dir, tag, str(num_train)), index=False)


def prepare_ppd_data():
    print(f"========================= prepare ppd data ============================ ")

    original_data_dir = ppd_data_creation['original_data_dir']
    to_dir = ppd_data_creation['processed_data_dir']

    data_all = original_data_dir + ppd_data_creation['original_ppd_data_file_name']
    data_datetime = original_data_dir + ppd_data_creation['original_ppd_datetime_file_name']

    df_data_all = pd.read_csv(data_all, skipinitialspace=True)
    df_data_datetime = pd.read_csv(data_datetime, skipinitialspace=True)

    print(f"[INFO] df_data_all: {df_data_all.shape}")
    print(f"[INFO] df_data_datetime: {df_data_datetime.shape}")
    # print(f"[INFO] 2015:{df_data_all[df_data_datetime['ListingInfo_Year'] == 2015].shape}")
    print(f"[INFO] 2014:{df_data_all[df_data_datetime['ListingInfo_Year'] == 2014].shape}")
    # print(f"[INFO] 2013:{df_data_all[df_data_datetime['ListingInfo_Year'] == 2013].shape}")
    # print(f"[INFO] 2012:{df_data_all[df_data_datetime['ListingInfo_Year'] == 2012].shape}")

    num_train = ppd_data_creation['number_train_samples']
    create_train_and_test(df_data_all, df_data_datetime, num_train, to_dir)


if __name__ == "__main__":
    prepare_ppd_data()
