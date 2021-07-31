import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data_process.census_process.mapping_resource import continuous_cols, categorical_cols, target_col_name
from datasets.census_dataset import SimpleDataset


def shuffle_data(data):
    len = data.shape[0]
    perm_idxs = np.random.permutation(len)
    return data[perm_idxs]


def get_datasets(ds_file_name, shuffle=False, split_ratio=0.9):
    dataframe = pd.read_csv(ds_file_name, skipinitialspace=True)

    COLUMNS_TO_LOAD = continuous_cols + categorical_cols + [target_col_name]
    print("[INFO] COLUMNS_TO_LOAD:", COLUMNS_TO_LOAD)
    samples = dataframe[COLUMNS_TO_LOAD].values
    # print(samples)
    if shuffle:
        samples = shuffle_data(samples)

    if split_ratio == 1.0:
        print(f"samples shape: {samples.shape}, {samples.dtype}")
        train_dataset = SimpleDataset(samples[:, :-1], samples[:, -1])
        return train_dataset, None
    else:
        num_train = int(split_ratio * samples.shape[0])
        train_samples = samples[:num_train].astype(np.float)
        val_samples = samples[num_train:].astype(np.float)
        print(f"train samples shape: {train_samples.shape}, {train_samples.dtype}")
        print(f"valid samples shape: {val_samples.shape}, {train_samples.dtype}")
        train_dataset = SimpleDataset(train_samples[:, :-1], train_samples[:, -1])
        val_dataset = SimpleDataset(val_samples[:, :-1], val_samples[:, -1])
        return train_dataset, val_dataset


def get_dataloaders(train_dataset: SimpleDataset, valid_dataset: SimpleDataset, batch_size=32, num_workers=1):
    mnist_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_valid_loader = None
    if valid_dataset is not None:
        mnist_valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=True, num_workers=num_workers)
    return mnist_train_loader, mnist_valid_loader


def get_income_census_dataloaders(ds_file_name, split_ratio=0.9, batch_size=64, num_workers=2):
    train_dataset, valid_dataset = get_datasets(ds_file_name=ds_file_name, shuffle=True, split_ratio=split_ratio)
    return get_dataloaders(train_dataset, valid_dataset, batch_size=batch_size, num_workers=num_workers)
