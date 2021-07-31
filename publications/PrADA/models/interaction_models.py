import os

import numpy as np
import torch
import torch.nn as nn

from models.classifier import TransformMatrix


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


def compute_interactive_key(idx_a, idx_b):
    return str(idx_a) + "-" + str(idx_b)


class BiFeatureInteractionComputer(object):

    def load_model(self, model_dict):
        return None

    def save_model(self, model_root, appendix):
        return None

    def freeze(self, is_freeze=False):
        pass

    def parameters(self):
        return list()

    def change_to_train_mode(self):
        pass

    def change_to_eval_mode(self):
        pass

    def build(self, feat_list):
        return feat_list

    def fit(self, feat_list):
        feat_intr_list = list()
        for i in range(len(feat_list) - 1):
            for j in range(i + 1, len(feat_list)):
                feat_intr_list.append(torch.cat((feat_list[i], feat_list[j]), dim=1))
        return feat_intr_list

    def build_fit(self, feat_list):
        return self.fit(feat_list)


# class AttentiveFeatureInteractionComputer(object):
#     def __init__(self, transform_matrix_dict=None):
#         self.transform_matrix_dict = transform_matrix_dict
#         self.score_map = None
#
#     def load_model(self, model_dict):
#         if self.transform_matrix_dict is None:
#             return
#
#         for key, model_path in model_dict.items():
#             self.transform_matrix_dict[key].load_state_dict(torch.load(model_path))
#             print(f"[INFO] load transform matrix model from {model_path}")
#
#     def save_model(self, model_root, appendix):
#         if self.transform_matrix_dict is None:
#             return None
#         transform_matrix_path_dict = dict()
#         for key, trans_model in self.transform_matrix_dict.items():
#             trans_model_name = "trans_model_" + key + "_" + appendix
#             trans_model_path = os.path.join(model_root, trans_model_name)
#             torch.save(trans_model.state_dict(), trans_model_path)
#             transform_matrix_path_dict[key] = trans_model_path
#             print(f"[INFO] saved transform matrix model {[trans_model_name]} to {trans_model_path}")
#         return transform_matrix_path_dict
#
#     def freeze(self, is_freeze=False):
#         if self.transform_matrix_dict is None:
#             return
#         for _, val in self.transform_matrix_dict.items():
#             for param in val.parameters():
#                 # print("AttentiveFeatureComputer requires_grad:", not is_freeze)
#                 param.requires_grad = not is_freeze
#
#     def parameters(self):
#         param_list = list()
#         if self.transform_matrix_dict is None:
#             return param_list
#         for _, val in self.transform_matrix_dict.items():
#             param_list += list(val.parameters())
#         return param_list
#
#     def change_to_train_mode(self):
#         if self.transform_matrix_dict is None:
#             return
#         for _, val in self.transform_matrix_dict.items():
#             val.train()
#
#     def change_to_eval_mode(self):
#         if self.transform_matrix_dict is None:
#             return
#         for _, val in self.transform_matrix_dict.items():
#             val.eval()
#
#     def transform(self, feat_a, idx_a, idx_b):
#         if self.transform_matrix_dict is None:
#             return feat_a
#
#         key = compute_interactive_key(idx_a, idx_b)
#         trans_matrix = self.transform_matrix_dict.get(key)
#         if trans_matrix:
#             t_feat_a = trans_matrix.transform(feat_a)
#         else:
#             key = compute_interactive_key(idx_b, idx_a)
#             trans_matrix = self.transform_matrix_dict[key]
#             t_feat_a = trans_matrix.transpose_transform(feat_a)
#         return t_feat_a
#
#     def compute_score(self, feat_a, feat_b, idx_a, idx_b):
#         t_feat_a = self.transform(feat_a, idx_a, idx_b)
#         # print("#t_feat_a", t_feat_a)
#         # print("#feat_b", feat_b)
#         output = torch.mul(t_feat_a, feat_b)
#         # print("#output", output)
#         return torch.sum(output, dim=1, keepdim=True)
#
#     def compute_score_map(self, feat_list):
#         score_map = dict()
#         for idx_a, feat_a in enumerate(feat_list):
#             for idx_b, feat_b in enumerate(feat_list):
#                 key = compute_interactive_key(idx_a, idx_b)
#                 score = self.compute_score(feat_a, feat_b, idx_a, idx_b)
#                 # print("#score1:", score)
#                 score = torch.exp(score)
#                 # print("#score2:", score)
#
#                 if score_map.get(key) is None:
#                     score_map[key] = score
#                     score_map[compute_interactive_key(idx_b, idx_a)] = score
#
#         return score_map
#
#     def compute_interactive_reprs(self, feat_list):
#         interactive_feat_list = list()
#         num_feat = len(feat_list)
#         for idx_a in range(num_feat):
#             # score_list = list()
#             interactive_feat = torch.zeros(feat_list[idx_a].shape)
#             all_score = torch.zeros((feat_list[idx_a].shape[0], 1))
#             for idx_b in range(0, num_feat):
#                 key = compute_interactive_key(idx_b, idx_a)
#                 score = self.score_map[key]
#                 all_score += score
#                 # score_list.append(score)
#                 trans_feat = self.transform(feat_list[idx_b], idx_b, idx_a)
#                 interactive_feat += score * trans_feat
#             interactive_feat = interactive_feat / all_score
#             interactive_feat_list.append(interactive_feat)
#         return interactive_feat_list
#
#     def build(self, feat_list):
#         self.score_map = self.compute_score_map(feat_list)
#         return self.score_map
#
#     def fit(self, feat_list):
#         return self.compute_interactive_reprs(feat_list)
#
#     def build_fit(self, feat_list):
#         self.build(feat_list)
#         return self.fit(feat_list)


class InteractionModel(object):

    def __init__(self, extractor_list, aggregator_list, discriminator_list, interactive_feature_computer):
        self.extractor_list = extractor_list
        self.aggregator_list = aggregator_list
        self.discriminator_list = discriminator_list
        self.interactive_feature_computer = interactive_feature_computer
        self.discriminator_criterion = nn.CrossEntropyLoss()
        self.discriminator_set = False if discriminator_list is None else True

    def load_model(self, model_dict):
        extractor_path_list = model_dict["extractor_path_list"]
        aggregator_path_list = model_dict["aggregator_path_list"]
        discriminator_path_list = model_dict["discriminator_path_list"]

        for extractor, path in zip(self.extractor_list, extractor_path_list):
            extractor.load_state_dict(torch.load(path))
            print(f"[INFO] load extractor model from {path}")

        for aggregator, path in zip(self.aggregator_list, aggregator_path_list):
            aggregator.load_state_dict(torch.load(path))
            print(f"[INFO] load aggregator model from {path}")

        for discriminator, path in zip(self.discriminator_list, discriminator_path_list):
            discriminator.load_state_dict(torch.load(path))
            print(f"[INFO] load discriminator model from {path}")

        self.interactive_feature_computer.load_model(model_dict['transform_matrix'])

    def save_model(self, model_root, appendix):
        extractor_path_list = list()
        extractor_name_prefix = "extractor_"
        for idx, extractor in enumerate(self.extractor_list):
            extractor_name = extractor_name_prefix + str(idx) + "_" + str(appendix)
            extractor_path = os.path.join(model_root, extractor_name)
            torch.save(extractor.state_dict(), extractor_path)
            extractor_path_list.append(extractor_path)
            print(f"[INFO] saved extractor model to: {extractor_path}")

        aggregator_path_list = list()
        aggregator_name_prefix = "aggregator_"
        for idx, aggregator in enumerate(self.aggregator_list):
            aggregator_name = aggregator_name_prefix + str(idx) + "_" + str(appendix)
            aggregator_path = os.path.join(model_root, aggregator_name)
            torch.save(aggregator.state_dict(), aggregator_path)
            aggregator_path_list.append(aggregator_path)
            print(f"[INFO] saved aggregator model to: {aggregator_path}")

        discriminator_path_list = list()
        discriminator_name_prefix = "discriminator_"
        for idx, discriminator in enumerate(self.discriminator_list):
            discriminator_name = discriminator_name_prefix + str(idx) + "_" + str(appendix)
            discriminator_path = os.path.join(model_root, discriminator_name)
            torch.save(discriminator.state_dict(), discriminator_path)
            discriminator_path_list.append(discriminator_path)
            print(f"[INFO] saved discriminator model to: {discriminator_path}")

        trans_matrix_meta = self.interactive_feature_computer.save_model(model_root, appendix)

        interactive_feature_meta = dict()
        interactive_feature_meta["extractor_path_list"] = extractor_path_list
        interactive_feature_meta["aggregator_path_list"] = aggregator_path_list
        interactive_feature_meta["discriminator_path_list"] = discriminator_path_list
        interactive_feature_meta["transform_matrix"] = trans_matrix_meta
        return interactive_feature_meta

    def check_discriminator_exists(self):
        if self.discriminator_set is False:
            raise RuntimeError('Discriminator not set.')

    def change_to_train_mode(self):
        if self.interactive_feature_computer:
            self.interactive_feature_computer.change_to_train_mode()

        for extractor, aggregator, discriminator in zip(self.extractor_list,
                                                        self.aggregator_list,
                                                        self.discriminator_list):
            extractor.train()
            aggregator.train()
            if self.discriminator_set:
                discriminator.train()

    def change_to_eval_mode(self):
        if self.interactive_feature_computer:
            self.interactive_feature_computer.change_to_eval_mode()

        for extractor, aggregator, discriminator in zip(self.extractor_list,
                                                        self.aggregator_list,
                                                        self.discriminator_list):
            extractor.eval()
            aggregator.eval()
            if self.discriminator_set:
                discriminator.eval()

    def parameters(self):
        param = self.extractor_parameters() + self.aggregator_parameters()
        if self.interactive_feature_computer:
            param += self.interactive_feature_computer.parameters()
        if self.discriminator_set:
            param += self.discriminator_parameters()
        return param

    def extractor_parameters(self):
        param_list = list()
        for extractor in self.extractor_list:
            param_list += list(extractor.parameters())
        return param_list

    def aggregator_parameters(self):
        param_list = list()
        for aggregator in self.aggregator_list:
            param_list += list(aggregator.parameters())
        return param_list

    def discriminator_parameters(self):
        param_list = list()
        for discriminator in self.discriminator_list:
            param_list += list(discriminator.parameters())
        return param_list

    def _freeze(self, model_list, is_freeze=False):
        for model in model_list:
            for param in model.parameters():
                param.requires_grad = not is_freeze

    def freeze(self, is_freeze=False):
        self._freeze(self.extractor_list, is_freeze)
        self._freeze(self.aggregator_list, is_freeze)
        self._freeze(self.discriminator_list, is_freeze)
        self.interactive_feature_computer.freeze(is_freeze)

    def get_num_feature_groups(self):
        return len(self.extractor_list)

    def _compute_fg_repr_list(self, fg_list):
        fg_repr_list = [extractor(fg) for fg, extractor in zip(fg_list, self.extractor_list)]
        return fg_repr_list

    def compute_output_list(self, fg_list):
        fg_intr_list = self.interactive_feature_computer.build_fit(fg_list)
        fg_intr_repr_list = self._compute_fg_repr_list(fg_intr_list)
        # fg_repr_int_list = self.interactive_feature_computer.build_fit(fg_repr_list)

        # fg_intr_list = self.int_feat_computer.build_fit(fg_list)
        # fg_repr_intr_list = self._compute_fg_repr_list(fg_intr_list)

        output_list = []
        for fg_repr_int, aggregator in zip(fg_intr_repr_list, self.aggregator_list):
            output_list.append(aggregator(fg_repr_int))

        return output_list

    def compute_loss(self, src_feat_list, tgt_feat_list, src_labels, tgt_labels, **kwargs):
        alpha = kwargs["alpha"]

        num_sample = src_feat_list[0].shape[0] + tgt_feat_list[0].shape[0]

        # src_fg_repr_list = list()
        # tgt_fg_repr_list = list()
        # for src_data, tgt_data, extractor in zip(src_data_list, tgt_data_list, self.extractor_list):
        #     src_fg_repr_list.append(extractor(src_data))
        #     tgt_fg_repr_list.append(extractor(tgt_data))
        #
        # src_intr_feat_list = self.interactive_feature_computer.build_fit(src_fg_repr_list)
        # tgt_intr_feat_list = self.interactive_feature_computer.build_fit(tgt_fg_repr_list)
        src_intr_feat_list = self.interactive_feature_computer.build_fit(src_feat_list)
        tgt_intr_feat_list = self.interactive_feature_computer.build_fit(tgt_feat_list)
        src_fg_repr_list = list()
        tgt_fg_repr_list = list()
        for src_data, tgt_data, extractor in zip(src_intr_feat_list, tgt_intr_feat_list, self.extractor_list):
            src_fg_repr_list.append(extractor(src_data))
            tgt_fg_repr_list.append(extractor(tgt_data))

        total_domain_loss = torch.tensor(0.)
        src_output_list = list()
        tgt_output_list = list()
        for src_feat, tgt_feat, aggregator, discriminator in zip(src_fg_repr_list, tgt_fg_repr_list,
                                                                 self.aggregator_list, self.discriminator_list):
        # for src_feat, tgt_feat, aggregator, discriminator in zip(src_intr_feat_list, tgt_intr_feat_list,
        #                                                          self.aggregator_list, self.discriminator_list):
            domain_feat = torch.cat((src_feat, tgt_feat), dim=0)
            domain_labels = torch.cat((src_labels, tgt_labels), dim=0)
            perm = torch.randperm(num_sample)
            domain_feat = domain_feat[perm]
            domain_labels = domain_labels[perm]

            src_output = aggregator(src_feat)
            tgt_output = aggregator(tgt_feat)
            domain_output = discriminator(domain_feat, alpha)
            domain_loss = self.discriminator_criterion(domain_output, domain_labels)

            src_output_list.append(src_output)
            tgt_output_list.append(tgt_output)
            total_domain_loss += domain_loss

        return total_domain_loss, src_output_list, tgt_output_list

    def calculate_domain_discriminator_correctness(self, fg_list, is_source=True):
        fg_intr_list = self.interactive_feature_computer.build_fit(fg_list)
        fg_intr_repr_list = self._compute_fg_repr_list(fg_intr_list)
        # fg_repr_list = self._compute_fg_repr_list(fg_list)
        # fg_repr_int_list = self.interactive_feature_computer.build_fit(fg_repr_list)

        corr_list = list()
        for fgi, extractor, discriminator in zip(fg_intr_repr_list, self.extractor_list, self.discriminator_list):
            if is_source:
                labels = torch.zeros(fgi.shape[0]).long()
            else:
                labels = torch.ones(fgi.shape[0]).long()
            # feat = extractor(fgi)
            pred = discriminator(fgi, alpha=0)
            pred_cls = pred.data.max(dim=1)[1]
            res = pred_cls.eq(labels).sum().item()
            corr_list.append(res)
        return corr_list
