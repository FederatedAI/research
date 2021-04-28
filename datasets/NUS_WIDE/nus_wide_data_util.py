import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# you need to put the data into following directory.
all_label_path = "NUS_WIDE/Groundtruth/AllLabels"
label_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
image_features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
text_tag_path = "NUS_WIDE/NUS_WID_Tags/"


def retrieve_top_k_labels(data_dir, top_k=10):
    """
        retrieve top k labels that occur most in all samples.

    Parameters
    ----------
    data_dir: the directory that stores NUS-WIDE data.
    top_k: the number of top labels to be returned.

    Returns
    -------
        the list of top k labels
    """
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, all_label_path)):
        file = os.path.join(data_dir, all_label_path, filename)
        # print(f"[INFO] load file:{file}")
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(os.path.join(file))
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    """
        Retrieve samples in the form of image features, textual features, and labels.

    Parameters
    ----------
    data_dir: the directory that stores NUS-WIDE data.
    selected_label: the list of labels of the samples to be retrieved.
    n_samples: the number of samples to be retrieved.
    dtype: the type of the samples to be retrieved (Train or Test), default "Train".

    Returns
    -------
        image data, text data, and labels in the form of one-hot vectors
    """

    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, label_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    if len(selected_label) > 1:
        selected_labels = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected_labels = data_labels
    print("[INFO] load rows with valid label", selected_labels.shape)

    dfs = []
    for file in os.listdir(os.path.join(data_dir, image_features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, image_features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            print(f"[INFO] load image feature ({file}) with ({len(df.columns)}) dimension.")
            dfs.append(df)
    all_image_features = pd.concat(dfs, axis=1)
    print(f"[INFO] load all image features with shape: {all_image_features.shape}")

    selected_image_features = all_image_features.loc[selected_labels.index]

    file = "_".join([dtype, "Tags1k"]) + ".dat"
    all_text_features = pd.read_csv(os.path.join(data_dir, text_tag_path, file), header=None, sep="\t")
    all_text_features.dropna(axis=1, inplace=True)
    print(f"[INFO] load all text features ({file}) with shape: {all_text_features.shape}.")

    selected_text_features = all_text_features.loc[selected_labels.index]

    if n_samples is None:
        return selected_image_features.values[:], selected_text_features.values[:], selected_labels.values[:]
    return selected_image_features.values[:n_samples], \
           selected_text_features.values[:n_samples], \
           selected_labels.values[:n_samples]


def get_image_and_text_data(data_dir, selected, n_samples=2000, dtype="Train"):
    return get_labeled_data(data_dir, selected, n_samples, dtype)


class TwoPartyNusWideDataLoader(object):

    def __init__(self, data_dir, binary_top_k_classes=5, binary_negative_label=0):
        """
        Parameters
        ----------
        data_dir: the directory where NUS-WIDE data located.
        binary_top_k_classes: load data of top k classes for binary classification.
        binary_negative_label: The integer that represents negative label.
        """
        self.data_dir = data_dir
        self.binary_top_k_classes = binary_top_k_classes
        self.binary_negative_label = binary_negative_label

    def get_train_data(self, target_labels, binary_classification=True, num_samples=None):
        return self.get_data(target_labels,
                             data_type="Train",
                             binary_classification=binary_classification,
                             num_samples=num_samples)

    def get_test_data(self, target_labels, binary_classification=True, num_samples=None):
        return self.get_data(target_labels,
                             data_type="Test",
                             binary_classification=binary_classification,
                             num_samples=num_samples)

    def get_data(self, target_labels, data_type, binary_classification=True, num_samples=None):
        if binary_classification:
            return self.load_data_with_binaryclasses(target_labels, self.data_dir, num_samples, dtype=data_type)
        else:
            return self.load_data_with_multiclasses(target_labels, self.data_dir, num_samples, dtype=data_type)

    def load_data_with_multiclasses(self, target_labels, data_dir, num_samples, dtype):
        if target_labels is not None and len(target_labels) <= 2:
            raise Exception(
                f"Multi-classification does not support the number of classes smaller than or equal to {len(target_labels)}")

        if target_labels is None:
            target_labels = retrieve_top_k_labels(data_dir, top_k=self.binary_top_k_classes)

        print(f"[INFO] load data with labels:{target_labels} for multi-classification.")
        image, text, labels = get_labeled_data(data_dir=data_dir,
                                               selected_label=target_labels,
                                               n_samples=num_samples,
                                               dtype=dtype)
        image, text, labels = shuffle(image, text, labels)
        return image, text, labels

    def load_data_with_binaryclasses(self, target_labels, data_dir, num_samples, dtype):
        if target_labels is None or len(target_labels) == 0:
            raise Exception("the label list must not be None or empty")
        elif len(target_labels) == 1:
            """
            if only one target label is set, then this target label will be treated as the positive label 
            and all other labels, which are top K labels, will be treated as negative labels.
            """
            top_k_labels = retrieve_top_k_labels(data_dir, top_k=self.binary_top_k_classes)
            if target_labels[0] in top_k_labels:
                top_k_labels.remove(target_labels[0])
                selected_labels = target_labels + top_k_labels
            else:
                selected_labels = target_labels + top_k_labels[:-1]

        elif len(target_labels) == 2:
            selected_labels = target_labels
        else:
            raise Exception(
                f"binary classification does not support {len(target_labels)} # of labels, which are {target_labels}.")

        print(f"[INFO] load data with labels:{selected_labels[0:1]} vs {selected_labels[1:]} "
              f"for binary-classification.")

        image, text, labels = get_labeled_data(data_dir=data_dir,
                                               selected_label=selected_labels,
                                               n_samples=num_samples,
                                               dtype=dtype)

        # change to binary labels
        binary_labels = []
        pos_count = 0
        neg_count = 0
        for i in range(labels.shape[0]):
            if labels[i, 0] == 1:
                binary_labels.append(1)
                pos_count += 1
            else:
                binary_labels.append(self.binary_negative_label)
                neg_count += 1

        print(f"[INFO] # of positive samples: {pos_count}, # of negative samples: {neg_count}")

        labels = np.array(binary_labels)
        image, text, labels = shuffle(image, text, labels)
        return image, text, labels
