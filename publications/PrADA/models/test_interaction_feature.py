from models.classifier import TransformMatrix
from models.interaction_models import AttentiveFeatureInteractionComputer, compute_interactive_key
import numpy as np
import torch


def initialize_transform_matrix(hidden_dim_a, hidden_dim_b):
    return TransformMatrix(hidden_dim_a, hidden_dim_b)


def initialize_transform_matrix_dict(hidden_dim_list):
    trans_matrix_dict = dict()
    for idx_a, hidden_dim_a in enumerate(hidden_dim_list):
        for idx_b, hidden_dim_b in enumerate(hidden_dim_list):
            transform_matrix = initialize_transform_matrix(hidden_dim_a, hidden_dim_b)
            key_1 = compute_interactive_key(idx_a, idx_b)
            key_2 = compute_interactive_key(idx_b, idx_a)
            if trans_matrix_dict.get(key_1) is None and trans_matrix_dict.get(key_2) is None:
                trans_matrix_dict[key_1] = transform_matrix
    return trans_matrix_dict


class TestInteractiveFeatureComputer(AttentiveFeatureInteractionComputer):

    def __init__(self, transform_matrix_dict):
        super(TestInteractiveFeatureComputer, self).__init__(transform_matrix_dict)

    def compute_score(self, feat_a, feat_b, idx_a, idx_b):
        return 1.0


if __name__ == "__main__":
    # list_1 = []
    # list_2 = [2, 3, 4]
    # list_3 = [5, 6, 7]
    # list = list_1 + list_2 + list_3
    # print(list)

    feat_1 = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]])
    feat_2 = np.array([[0.4, 0.5, 0.6, 0.7, 0.8],
                       [0.14, 0.15, 0.16, 0.17, 0.18]])
    feat_3 = np.array([[0.71, 0.8, 0.9, 0.10],
                       [0.24, 0.25, 0.26, 0.27]])
    feat_list = [feat_1, feat_2, feat_3]

    print("feat_list:", feat_list)

    hidden_dim_list = [f.shape[-1] for f in feat_list]
    print("hidden_dim_list:", hidden_dim_list)

    transform_matrix_dict = initialize_transform_matrix_dict(hidden_dim_list)
    print("transform_matrix_dict: \n", transform_matrix_dict)

    computer = AttentiveFeatureInteractionComputer(transform_matrix_dict)

    feat_tensor_list = [torch.tensor(f, dtype=torch.float) for f in feat_list]

    score_map = computer.build(feat_tensor_list)
    print("score_map: \n", score_map)

    int_feat_list = computer.fit(feat_tensor_list)
    print("int_feat_list: \n")
    for int_feat in int_feat_list:
        print(int_feat)

    print()
    print("parameter: \n", computer.parameters())
