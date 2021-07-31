import json
import os
from datetime import date
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, accuracy_score

from statistics_utils import kl_divergence


def compute_parameter_size(feature_extractor_architecture):
    all_num_param = 0
    for archi in feature_extractor_architecture:
        for i in range(1, len(archi)):
            all_num_param += archi[i] * archi[i - 1]
        print(f"{archi} has # of parameters:{all_num_param}")
    return all_num_param


def create_id_from_hyperparameters(hyper_parameter_dict):
    hyper_param_list = [key + str(value) for key, value in hyper_parameter_dict.items()]
    return "_".join(hyper_param_list)


def get_latest_timestamp(timestamped_file_name, folder):
    timestamp_list = []
    for filename in os.listdir(folder):
        if filename.startswith(timestamped_file_name):
            maybe_timestamp = filename.split("_")[-1]
            # print("[DEBUG] [get_latest_timestamp()] maybe_timestamp: ", maybe_timestamp)
            if maybe_timestamp.endswith(".json"):
                timestamp = int(maybe_timestamp.split(".")[0])
            else:
                timestamp = int(maybe_timestamp)
            timestamp_list.append(timestamp)
    timestamp_list.sort()
    latest_timestamp = timestamp_list[-1]
    return latest_timestamp


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def get_current_date():
    return date.today().strftime("%Y%m%d")


def save_dann_experiment_result(root, task_id, param_dict, metric_dict, timestamp):
    task_folder = task_id
    task_root_folder = os.path.join(root, task_folder)
    if not os.path.exists(task_root_folder):
        os.makedirs(task_root_folder)

    result_dict = dict()
    result_dict["lr_param"] = param_dict
    result_dict["metrics"] = metric_dict

    file_name = "dann_exp_result_" + str(timestamp) + '.json'
    file_full_name = os.path.join(task_root_folder, file_name)
    with open(file_full_name, 'w') as outfile:
        json.dump(result_dict, outfile)


def load_dann_experiment_result(root, task_id, timestamp=None):
    task_folder = "task_" + task_id
    task_folder_path = os.path.join(root, task_folder)
    if not os.path.exists(task_folder_path):
        raise FileNotFoundError(f"{task_folder_path} is not found.")

    experiment_result = "dann_exp_result"
    if timestamp is None:
        timestamp = get_latest_timestamp(experiment_result, task_folder_path)
        print(f"[INFO] get latest timestamp {timestamp}")

    experiment_result_file_name = str(experiment_result) + "_" + str(timestamp) + '.json'
    experiment_result_file_path = os.path.join(task_folder_path, experiment_result_file_name)
    if not os.path.exists(experiment_result_file_path):
        raise FileNotFoundError(f"{experiment_result_file_path} is not found.")

    with open(experiment_result_file_path) as json_file:
        print(f"[INFO] load experiment result file from {experiment_result_file_path}")
        dann_exp_result_dict = json.load(json_file)
    return dann_exp_result_dict


def test_classifier(model, data_loader, tag):
    print(f"---------- {tag} classification ----------")
    correct = 0
    n_total = 0
    y_pred_list = []
    y_real_list = []
    y_pos_pred_prob_list = []
    model.change_to_eval_mode()
    for batch_idx, (data, label) in enumerate(data_loader):
        label = label.flatten()
        n_total += len(label)
        batch_corr, y_pred, pos_y_prob = model.calculate_classifier_correctness(data, label)
        correct += batch_corr
        y_real_list += label.tolist()
        y_pred_list += y_pred.tolist()
        y_pos_pred_prob_list += pos_y_prob.tolist()

    acc = correct / n_total
    auc_0 = roc_auc_score(y_real_list, y_pred_list)
    auc_1 = roc_auc_score(y_real_list, y_pos_pred_prob_list)

    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks = get_ks(np.array(y_pos_pred_prob_list), np.array(y_real_list))
    print("[INFO]: {}/{}".format(correct, n_total))
    # print("[INFO]: roc_auc_score_0 : ", auc_0)
    print("[INFO]: test AUC : ", auc_1)
    print("[INFO]: test KS  : ", ks)
    return acc, auc_1, ks


def test_discriminator(model, num_regions, source_loader, target_loader):
    print("---------- test_discriminator ----------")
    source_correct = np.zeros(num_regions)
    target_correct = np.zeros(num_regions)
    n_source_total = 0
    n_target_total = 0

    for source_batch in source_loader:
        source_data, source_label = source_batch
        n_source_total += len(source_label)
        src_corr_lst = model.calculate_domain_discriminator_correctness(source_data, is_source=True)
        source_correct += np.array(src_corr_lst)

    for target_batch in target_loader:
        target_data, target_label = target_batch
        n_target_total += len(target_label)
        tgt_corr_lst = model.calculate_domain_discriminator_correctness(target_data, is_source=False)
        target_correct += np.array(tgt_corr_lst)

    total_acc = (source_correct + target_correct) / (n_source_total + n_target_total)
    source_acc = source_correct / n_source_total
    target_acc = target_correct / n_target_total
    cat_acc = np.concatenate((source_acc.reshape(1, -1), target_acc.reshape(1, -1)), axis=0)
    acc_sum = np.sum(cat_acc, axis=0)
    print(f"normalized domain acc:\n {cat_acc / acc_sum}")

    # overall_acc = (source_correct + target_correct)
    ave_total_acc = np.mean(total_acc)
    ave_source_acc = np.mean(source_acc)
    ave_target_acc = np.mean(target_acc)
    print(f"[DEBUG] {n_source_total} source domain acc: {source_acc}, mean: {ave_source_acc}")
    print(f"[DEBUG] {n_target_total} target domain acc: {target_acc}, mean: {ave_target_acc}")
    print(f"[DEBUG] total domain acc: {total_acc}, mean: {ave_total_acc}")
    entropy_domain_acc = entropy(cat_acc / acc_sum)
    ave_entropy = np.mean(entropy_domain_acc)
    print(f"[DEBUG] domain acc entropy: {entropy_domain_acc}, mean:{ave_entropy} ")
    return (ave_total_acc, ave_source_acc, ave_target_acc), (
        list(total_acc), list(source_acc), list(target_acc)), ave_entropy


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * np.log2(predictions + epsilon)
    return H.sum(axis=0)


def compute(auc_list, ks_list, tag, top_k=5):
    print("-" * 150)
    ppd_fg_sum = np.array(auc_list) + np.array(ks_list)
    sorted_result = np.argsort(ppd_fg_sum)
    ppd_fg_top5_auc = auc_list[sorted_result[-top_k:]]
    ppd_fg_top5_ks = ks_list[sorted_result[-top_k:]]
    print("[INFO] {}_top5_auc:{}, mean:{}, std:{}".format(tag, ppd_fg_top5_auc, np.mean(ppd_fg_top5_auc),
                                                          np.std(ppd_fg_top5_auc)))
    print("[INFO] {}_top5_ks:{}, mean:{}, std:{}".format(tag, ppd_fg_top5_ks, np.mean(ppd_fg_top5_ks),
                                                         np.std(ppd_fg_top5_ks)))


def produce_data_for_lr_shap(model, data_loader, column_name_list, output_file_full_name):
    """
    produce data for LR SHAP
    """

    sample_list = []
    for data, label in data_loader:
        feature = model.calculate_global_classifier_input_vector(data).detach().numpy()
        label = label.numpy().reshape((-1, 1))
        # print(feature.shape, label.shape)
        sample = np.concatenate((feature, label), axis=1)
        sample_list.append(sample)
    classifier_data = np.concatenate(sample_list, axis=0)

    print(f"[INFO] global classifier input data with shape:{classifier_data.shape}")
    df_lr_input = pd.DataFrame(data=classifier_data, columns=column_name_list)
    # print(df_lr_input.head(5))
    df_lr_input.to_csv(output_file_full_name, index=False)
    print(f"[INFO] save data to {output_file_full_name}")


def produce_data_for_distribution(model,
                                  src_train_loader,
                                  tgt_train_loader,
                                  feature_group_name_list,
                                  to_dir,
                                  tag,
                                  version="0"):
    src_train_iter = iter(src_train_loader)
    tgt_train_iter = iter(tgt_train_loader)

    src_data, _ = src_train_iter.next()
    tgt_data, _ = tgt_train_iter.next()

    src_fg_emb_list = model.calculate_feature_group_embedding_list(src_data)
    tgt_fg_emb_list = model.calculate_feature_group_embedding_list(tgt_data)

    for src_fg_emb, tgt_fg_emb, name in zip(src_fg_emb_list, tgt_fg_emb_list, feature_group_name_list):
        df_src_data = pd.DataFrame(data=src_fg_emb.detach().numpy())
        df_tgt_data = pd.DataFrame(data=tgt_fg_emb.detach().numpy())
        file_fill_name = to_dir + "/prada_{}_src_emb_{}_v{}.csv".format(tag, name, version)
        df_src_data.to_csv(file_fill_name, header=False, index=False)
        print(f"save df_src_data to {file_fill_name} with shape {df_src_data.shape}")

        file_fill_name = to_dir + "/prada_{}_tgt_emb_{}_v{}.csv".format(tag, name, version)
        df_tgt_data.to_csv(file_fill_name, header=False, index=False)
        print(f"save df_tgt_data to {file_fill_name} with shape {df_tgt_data.shape}")


def compute_kl_divergence(src_data, tgt_data, n_components):
    kl = 0.0
    for idx in range(n_components):
        kl += kl_divergence(tgt_data[:, idx], src_data[:, idx])
    kl = kl / n_components
    print(f"[INFO] KL-divergence: {kl}")
    return kl


def draw_distribution(df_src_data, df_tgt_data, num_points, dim_reducer, tag, feature_group_name, to_dir, version):
    print("[INFO] draw distribution")

    df_src_tgt_data = pd.concat([df_src_data, df_tgt_data], axis=0)
    num_src = num_points if df_src_data.shape[0] > num_points else df_src_data.shape[0]

    all_data = dim_reducer.fit_transform(df_src_tgt_data.values)

    # TODO: save all data
    print(f"[INFO] src_data shape:{df_src_data.shape}")
    print(f"[INFO] tgt_data shape:{df_tgt_data.shape}")
    print(f"[INFO] all_data shape:{all_data.shape}")
    print(f"[INFO] KL:{dim_reducer.kl_divergence_}")

    # estimate KL divergence
    kl = compute_kl_divergence(src_data=all_data[:num_src],
                               tgt_data=all_data[-num_src:],
                               n_components=all_data.shape[1])

    plt.figure(figsize=(8, 8))
    plt.scatter(all_data[:num_src, 0], all_data[:num_src, 1], s=10, c='r')
    plt.scatter(all_data[-num_src:, 0], all_data[-num_src:, 1], s=10, c='b')
    plt.title(tag + "-" + feature_group_name)
    plt.xlabel(f"KL={kl}")
    file_full_name = to_dir + "distr_{}_{}_{}_v{}.png".format(tag, feature_group_name, num_src, version)
    plt.savefig(file_full_name)
    print(f"[INFO] save fig to {file_full_name}")
    # plt.show()
