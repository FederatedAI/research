import csv
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.utils import shuffle

#from data_util.data_loader import TwoPartyDataLoader


def balance_X_y(XA, XB, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    # num_neg = np.sum(y == -1)
    # pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    # neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0]

    num_neg = np.sum(y == 0)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0.5]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0.5]

    print("len(pos_indexes)", len(pos_indexes))
    print("len(neg_indexes)", len(neg_indexes))
    print("num of samples", len(pos_indexes) + len(neg_indexes))
    print("num_pos:", num_pos)
    print("num_neg:", num_neg)

    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        # randomly pick negative samples of size equal to that of positive samples
        rand_indexes = neg_indexes[:num_pos]
        indexes = pos_indexes + rand_indexes
        np.random.shuffle(indexes)
        y = [y[i] for i in indexes]
        XA = [XA[i] for i in indexes]
        XB = [XB[i] for i in indexes]

    return np.array(XA), np.array(XB), np.array(y)


def get_top_k_labels(data_dir, top_k=5):
    data_path = "NUS_WIDE/Groundtruth/AllLabels"
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        # print(file)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(os.path.join(data_dir, file))
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        print("df shape", df.shape)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    print(selected.shape)

    # get XA, which are image low level features
    features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
    print("data_dir: {0}".format(data_dir))
    print("features_path: {0}".format(features_path))
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            print("b datasets features", len(df.columns))
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    print("X image shape:", data_X_image_selected.shape)  # 634 columns

    # get XB, which are tags
    tag_path = "NUS_WIDE/NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    print("X text shape:", data_X_text_selected.shape)

    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], selected.values[:]
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], selected.values[:n_samples]


def image_and_text_data(data_dir, selected, n_samples=2000):
    return get_labeled_data(data_dir, selected, n_samples)


# class TwoPartyNusWideDataLoader(TwoPartyDataLoader):

#     def __init__(self, infile):
#         self.infile = infile
#         self.labels = None
#         self.binary_top_k_classes = 10

#     def set_target_labels(self, labels):
#         self.labels = labels

#     # def get_training_data(self, num_samples=None):
#     #     if len(self.labels) == 1:
#     #         return self.load_data(self.infile, num_samples)
#     #     elif len(self.labels) > 1:
#     #         return self.load_data_with_multiclass(self.infile, num_samples)
#     #     else:
#     #         raise Exception("target labels should be more than zero, but get {0}".format(self.labels))

#     def get_training_data(self, binary_classification=True, num_samples=None):
#         if binary_classification:
#             return self.load_data(self.infile, num_samples)
#         else:
#             return self.load_data_with_multiclass(self.infile, num_samples)

#     def load_data_with_multiclass(self, infile, num_samples):
#         X_image, X_text, y_one_hot = get_labeled_data(data_dir=infile,
#                                                       selected_label=self.labels,
#                                                       n_samples=num_samples)
#         print("X_image shape:", X_image.shape)
#         print("X_text shape:", X_text.shape)
#         print("y_one_hot shape", y_one_hot.shape)

#         X_image, X_text, y = shuffle(X_image, X_text, y_one_hot)
#         print("X_image shape:", X_image.shape)
#         print("X_text shape:", X_text.shape)
#         print("y shape:", y.shape)
#         return X_image, X_text, y

#     def load_data(self, infile, num_samples):

#         sel = self.labels
#         if len(self.labels) == 1:
#             sel = self.labels
#             all_labels = get_top_k_labels(infile, top_k=self.binary_top_k_classes)
#             all_labels.remove(self.labels[0])
#             sel = sel + all_labels
#         elif len(self.labels) == 0:
#             raise Exception("label list should not be empty")
#         print("sel:", sel)

#         X_image, X_text, y = get_labeled_data(data_dir=infile, selected_label=sel, n_samples=num_samples)
#         print("X_image shape:", X_image.shape)
#         print("X_text shape:", X_text.shape)
#         print("y shape", y.shape)

#         y_ = []
#         pos_count = 0
#         neg_count = 0
#         for i in range(y.shape[0]):
#             if y[i, 0] == 1:
#                 y_.append(1)
#                 pos_count += 1
#             else:
#                 y_.append(0)
#                 neg_count += 1

#         print("pos counts:", pos_count)
#         print("neg counts:", neg_count)

#         y_ = np.array(y_)
#         # X_image, X_text, y_ = balance_X_y(X_image, X_text, y_)
#         X_image, X_text, y = shuffle(X_image, X_text, y_)
#         print("X_image shape:", X_image.shape)
#         print("X_text shape:", X_text.shape)
#         return X_image, X_text, y


def get_images():
    image_urls = "D:/Data/NUS_WIDE/NUS_WIDE/NUS-WIDE-urls/NUS-WIDE-urls.txt"
    # df = pd.read_csv(image_urls, header=0, sep=" ")
    # print(df.head(10))
    # kkk = df.loc[:, "url_Middle"]
    # print(kkk.head(10))

    read_num_urls = 1
    with open(image_urls, "r") as fi:
        fi.readline()
        reader = csv.reader(fi, delimiter=' ', skipinitialspace=True)
        for idx, row in enumerate(reader):
            if idx >= read_num_urls:
                break
            print(row[0], row[2], row[3], row[4])
            if row[3] is not None and row[3] != "null":
                url = row[4]
                print("{0} url: {1}".format(idx, url))

                str_array = row[0].split("\\")
                print(str_array[3], str_array[4])

                # img = imageio.imread(url)
                # print(type(img), img.shape)

                response = requests.get(url)
                print(response.status_code)
                img = Image.open(BytesIO(response.content))
                arr = np.array(img)
                print(type(img), arr.shape)
                # imageio.imwrite("", img)
                size = 48, 48
                img.thumbnail(size)
                img.show()
                arr = np.array(img)
                print("thumbnail", arr.shape)


if __name__ == '__main__':
    get_images()
