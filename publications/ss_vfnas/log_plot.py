# -*- coding:utf-8 -*-
# @Author   : LuoJiahuan
# @File     : log_plot.py 
# @Time     : 2020/9/10 10:30

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x_list = [11.2, 21.3, 23.642, 0.77, 1.320, 1.05, 2.5, 4.1]

# y_list = [82.5625, 82.1875, 81.375, 78.9375, 79.000, 81.4375, 82.3125, 82.000, 80.5625]
y_list = [81.8750, 81.3125, 81.375, 78.9375, 79.000, 81.4375, 82.3125, 82.000]

plt.figure()
ax = plt.gca()
ax.set_title("Test Accuracy vs. Model size over 2 parties")
ax.set_xlabel("Model size", fontsize=14)
ax.set_ylabel("Test Accuracy(%)", fontsize=14)

# ax.scatter(x_list[0], y_list[0], c='g', s=50, alpha=1, marker='v')
# ax.scatter(x_list[1], y_list[1], c='g', s=80, alpha=1, marker='^')
# ax.scatter(x_list[2], y_list[2], c='g', s=100, alpha=1, marker='<')
# ax.scatter(x_list[3], y_list[3], c='g', s=35, alpha=1, marker='o')
# ax.scatter(x_list[4], y_list[4], c='r', s=40, alpha=1, marker='o')
# ax.scatter(x_list[5], y_list[5], c='b', s=40, alpha=1, marker='o')
# ax.scatter(x_list[6], y_list[6], c='b', s=80, alpha=1, marker='o')
# ax.scatter(x_list[7], y_list[7], c='b', s=120, alpha=1, marker='o')

ax.scatter(x_list[0], y_list[0], c='g', s=80, alpha=1, marker='s')
ax.scatter(x_list[1], y_list[1], c='g', s=80, alpha=1, marker='s')
ax.scatter(x_list[2], y_list[2], c='g', s=80, alpha=1, marker='s')
ax.scatter(x_list[3], y_list[3], c='blueviolet', s=80, alpha=1, marker='X')
ax.scatter(x_list[4], y_list[4], c='r', s=80, alpha=1, marker='h')
ax.scatter(x_list[5], y_list[5], c='b', s=80, alpha=1, marker='o')
ax.scatter(x_list[6], y_list[6], c='b', s=80, alpha=1, marker='o')
ax.scatter(x_list[7], y_list[7], c='b', s=80, alpha=1, marker='o')

# ax.scatter(x_list[8], y_list[8], c='b', s=160, alpha=1, marker='o')
plt.legend(
    ["ResNet18", "ResNet34", "ResNet50", "SqueezeNet", "ShuffleNet V2", "DARTS-S", "DARTS-M", "DARTS-L"])
plt.savefig("eval/acc_size.png", bbox_inches='tight', dpi=320)

x_list = [1, 2, 3, 4, 5, 6]

darts = [79.6250, 82.3125, 83.1875, 84.3125, 84.5625, 85.2500]
mile = [80.2500, 82.3125, 83.6245, 83.9375, 84.2500, 85.6250]
ssnas = [80.25,83.0625,83.25,84.9375,]
shufflenet = [76.3750, 79.0000, 80.5000, 80.9375, 80.5625, 80.8125]
squeezenet = [75.4375, 78.8750, 78.6250, 79.3125, 79.5600, 80.6250]
res18 = [79.9375, 81.8750, 83.3750, 83.3750, 83.6250, 83.9375]
res34 = [80.5000, 81.3125, 82.3125, 83.5000, 83.8125, 84.3125]
res50 = [79.3125, 80.6875, 81.8125, 82.0625, 82.8750, 83.0000]
plt.figure()
ax = plt.gca()
# ax.set_title("Test acc vs. num of parties")
ax.set_xlabel("Number of Parties", fontsize=14)
ax.set_ylabel("Test Accuracy(%)", fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(x_list, darts, 'o', linestyle='--', linewidth=2.0, label='DARTS')
plt.plot(x_list, mile, 'o', linestyle='--', linewidth=2.0, label='MiLeNAS')
plt.plot(x_list, shufflenet, 'o', linestyle='--', linewidth=2.0, label='ShuffleNet V2')
plt.plot(x_list, squeezenet, 'o', linestyle='--', linewidth=2.0, label='SqueezeNet')
plt.plot(x_list, res18, 'o', linestyle='--', linewidth=2.0, label='ResNet18')
plt.plot(x_list, res34, 'o', linestyle='--', linewidth=2.0, label='ResNet34')
plt.plot(x_list, res50, 'o', linestyle='--', linewidth=2.0, label='ResNet50')

plt.grid(linestyle='-.')

# dodgerblue
# ax.scatter(x_list[0], y_list[0], s=30, c='dodgerblue', alpha=1, marker='o')
# ax.scatter(x_list[1], y_list[1], s=30, c='dodgerblue', alpha=1, marker='o')
# ax.scatter(x_list[2], y_list[2], s=30, c='dodgerblue', alpha=1, marker='o')
# ax.scatter(x_list[3], y_list[3], s=30, c='dodgerblue', alpha=1, marker='o')
# ax.scatter(x_list[4], y_list[4], s=30, c='dodgerblue', alpha=1, marker='o')
# ax.scatter(x_list[5], y_list[5], s=30, c='dodgerblue', alpha=1, marker='o')
# plt.legend(["1", "2", "3", "4"])
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, fontsize=10)

plt.savefig("eval/diff_party.png", bbox_inches='tight', dpi=320)

darts = list(map(lambda n: n - 79.6250, darts))
mile = list(map(lambda n: n - 80.2500, mile))
shufflenet = list(map(lambda n: n - 76.3750, shufflenet))
squeezenet = list(map(lambda n: n - 75.4375, squeezenet))
res18 = list(map(lambda n: n - 79.9375, res18))
res34 = list(map(lambda n: n - 80.5000, res34))
res50 = list(map(lambda n: n - 79.3125, res50))
for i in range(len(darts)):
    darts[i] = darts[i] / (i + 1)
    mile[i] = mile[i] / (i + 1)
    shufflenet[i] = shufflenet[i] / (i + 1)
    squeezenet[i] = squeezenet[i] / (i + 1)
    res18[i] = res18[i] / (i + 1)
    res34[i] = res34[i] / (i + 1)
    res50[i] = res50[i] / (i + 1)

plt.figure()
ax = plt.gca()
# ax.set_title("Test acc vs. num of parties")
ax.set_xlabel("Number of Parties", fontsize=14)
ax.set_ylabel("Improvement ratio", fontsize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(x_list, darts, 'o', linestyle='--', linewidth=2.0, label='DARTS')
plt.plot(x_list, mile, 'o', linestyle='--', linewidth=2.0, label='MiLeNAS')
plt.plot(x_list, shufflenet, 'o', linestyle='--', linewidth=2.0, label='ShuffleNet V2')
plt.plot(x_list, squeezenet, 'o', linestyle='--', linewidth=2.0, label='SqueezeNet')
plt.plot(x_list, res18, 'o', linestyle='--', linewidth=2.0, label='ResNet18')
plt.plot(x_list, res34, 'o', linestyle='--', linewidth=2.0, label='ResNet34')
plt.plot(x_list, res50, 'o', linestyle='--', linewidth=2.0, label='ResNet50')

plt.grid(linestyle='-.')
plt.legend(bbox_to_anchor=(1.01, 0), loc=3, fontsize=10)

plt.savefig("eval/improve_ratio.png", bbox_inches='tight', dpi=320)


# log_name = "proximal_reg"
#
# log_path = osp.join("search", log_name, log_name + ".log")
# csv_path = osp.join("search", log_name, log_name + ".csv")
# png_path = osp.join("search", log_name, log_name + ".png")
# log_file = open(log_path).readlines()
# train_prec = []
# val_prec = []
# for line in log_file:
#     if "Final" in line:
#         prec = line.strip().split()[-1][:-1]
#         if "Train" in line:
#             train_prec.append(float(prec))
#         elif "Valid" in line:
#             val_prec.append(float(prec))
#         # else:
#         #     raise ValueError
#
# with open(csv_path, "w") as f:
#     for i in range(len(train_prec)):
#         f.write(str(train_prec[i]))
#         f.write(",")
#         f.write(str(val_prec[i]))
#         f.write("\n")
#     f.close()
#
# x_train = np.linspace(0, len(train_prec), len(train_prec))
# y_train = np.array(train_prec)
#
# y_val = np.array(val_prec)
#
# plt.figure()
# plt.plot(x_train, y_train, label="Train Accuracy")
# plt.plot(x_train, y_val, label="Valid Accuracy")
# legend = plt.legend(loc=4)
# legend.get_title().set_fontsize(fontsize=14)
#
# plt.title(log_name, fontsize=14)
# plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("Accuracy", fontsize=14)
# plt.savefig(png_path)

#
# baseline = open(osp.join("search", "baseline", "log.txt")).readlines()
# darts_baeline_only_A = open(osp.join("search", "darts_baeline_only_A", "darts_baeline_only_A" + ".log")).readlines()
# milenas_baeline_only_A = open(
#     osp.join("search", "milenas_baeline_only_A", "milenas_baeline_only_A" + ".log")).readlines()
# milenas_baseline = open(osp.join("search", "milenas_baseline", "milenas_baseline" + ".log")).readlines()
#
# baseline_y = []
#
# darts_baeline_only_A_x = []
# darts_baeline_only_A_y = []
#
# milenas_baeline_only_A_x = []
# milenas_baeline_only_A_y = []
#
# milenas_baseline_x = []
# milenas_baseline_y = []
#
#
# for line in baseline:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         baseline_y.append(float(prec))
#
# for line in darts_baeline_only_A:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         darts_baeline_only_A_y.append(float(prec))
#
# for line in milenas_baeline_only_A:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         milenas_baeline_only_A_y.append(float(prec))
#
# for line in milenas_baseline:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         milenas_baseline_y.append(float(prec))
#
# multiview_x = np.linspace(0, len(baseline_y), len(baseline_y))
# darts_baeline_only_A_x = np.linspace(0, len(darts_baeline_only_A_y), len(darts_baeline_only_A_y))
# milenas_baeline_only_A_x = np.linspace(0, len(milenas_baeline_only_A_y), len(milenas_baeline_only_A_y))
# milenas_baseline_x = np.linspace(0, len(milenas_baseline_y), len(milenas_baseline_y))
# x_new = np.arange(0, multiview_x[-1], 0.1)
# func = interpolate.interp1d(multiview_x, baseline_y, kind='cubic')
# baseline_y_new = func(x_new)
# plt.figure()
# plt.plot(x_new, baseline_y_new, label="VFL_DARTS", linewidth='1.8')
# # plt.plot(milenas_baseline_x, milenas_baseline_y, label="VFL_MiLeNAS", linestyle='-', linewidth='1.8')
# # plt.plot(darts_baeline_only_A_x, darts_baeline_only_A_y, label="Vanilla_DARTS", linewidth='1.8')
# # plt.plot(milenas_baeline_only_A_x, milenas_baeline_only_A_y, label="Vanilla_MiLeNAS", linestyle='-', linewidth='1.8',
# #          color='g')
# plt.legend(loc=4)
#
# # plt.title("Val_acc (%)")
# plt.xlabel("Epoch")
# plt.ylabel("Val_acc (%)")
# plt.savefig("search/milenas_baeline_only_A.png")


def commu_effe():
    one_party_darts = open(osp.join("search", "1_party_darts-20200916-142539", "log.txt")).readlines()
    one_party_milenas = open(osp.join("search", "1_party_milenas-20200914-114257", "log.txt")).readlines()
    two_party_darts = open(osp.join("search", "2_party_darts-20200916-120940", "log.txt")).readlines()
    two_party_milenas = open(osp.join("search", "2_party_milenas-20200914-114331", "log.txt")).readlines()

    one_party_darts_y = []

    one_party_milenas_y = []

    two_party_darts_y = []

    two_party_milenas_y = []

    for line in one_party_darts:
        if "valid_acc" in line:
            prec = line.strip().split()[-1]
            one_party_darts_y.append(float(prec))

    for line in one_party_milenas:
        if "valid_acc" in line:
            prec = line.strip().split()[-1]
            one_party_milenas_y.append(float(prec))

    for line in two_party_darts:
        if "valid_acc" in line:
            prec = line.strip().split()[-1]
            two_party_darts_y.append(float(prec))

    for line in two_party_milenas:
        if "valid_acc" in line:
            prec = line.strip().split()[-1]
            two_party_milenas_y.append(float(prec))
    x = np.linspace(0, len(one_party_darts_y[:50]) * 198 * 2, len(one_party_darts_y[:50]))
    x_1 = np.linspace(0, len(one_party_darts_y[:50]) * 198, len(one_party_darts_y[:50]))
    multiview_x = np.arange(0, len(one_party_darts_y[:50]) * 198 * 2, 198 * 2)
    multiview_x_1 = np.arange(0, len(one_party_darts_y[:50]) * 198, 198)
    # darts_baeline_only_A_x = np.arange(0, len(darts_baeline_only_A_y), 0.1)
    # milenas_baeline_only_A_x = np.arange(0, len(milenas_baeline_only_A_y), 0.1)
    # milenas_baseline_x = np.arange(0, len(milenas_baseline_y), 0.1)
    one_party_darts = interpolate.interp1d(x, one_party_darts_y[:50], kind='cubic')(multiview_x)
    one_party_milenas = interpolate.interp1d(x_1, one_party_milenas_y[:50], kind='cubic')(multiview_x_1)
    two_party_darts = interpolate.interp1d(x, two_party_darts_y[:50], kind='cubic')(multiview_x)
    two_party_milenas = interpolate.interp1d(x_1, two_party_milenas_y[:50], kind='cubic')(multiview_x_1)
    plt.figure(figsize=(10, 8), dpi=320)
    # plt.plot(multiview_x, one_party_darts, label="1_Party_DARTS", linewidth='1.8')
    # plt.plot(multiview_x_1, one_party_milenas, label="1_Party_MiLeNAS", linestyle='-', linewidth='1.8')

    ax = plt.gca()
    # ax.set_title("Test acc vs. num of parties")
    ax.set_xlabel("Number of Parties", fontsize=14)
    ax.set_ylabel("Improvement ratio", fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.plot(multiview_x, two_party_darts, label="2_Parties_DARTS", linewidth='1.8')
    plt.plot(multiview_x_1, two_party_milenas, label="2_Parties_MiLeNAS", linewidth='1.8', )
    plt.tick_params(labelsize=14)
    plt.legend(loc=4)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=16)
    # plt.title("Val_acc (%)")
    plt.xlabel("Communication Round", fontsize=14)
    plt.ylabel("Test_acc (%)", fontsize=14)
    # plt.xlabel("Communication Round", fontsize=14)
    # plt.ylabel("Val_Acc", fontsize=14)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.subplots_adjust(hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig("search/commu_eff.png", bbox_inches='tight', dpi=320)


# baseline = open(osp.join("search", "baseline", "log.txt")).readlines()
# darts_proximal_A = open(osp.join("search", "darts_proximal_A", "darts_proximal_A" + ".log")).readlines()
#
# baseline_y = []
#
# darts_proximal_A_y = []
#
# for line in baseline:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         baseline_y.append(float(prec))
#
# for line in darts_baeline_only_A:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         darts_proximal_A_y.append(float(prec))
#
# baseline_x = np.linspace(0, len(baseline_y), len(baseline_y))
# darts_proximal_A_x = np.linspace(0, len(darts_proximal_A_y), len(darts_proximal_A_y))
#
# plt.figure()
# plt.plot(multiview_x, baseline_y, label="VFL_DARTS", linewidth='1.8')
# plt.plot(milenas_baseline_x, milenas_baseline_y, label="VFL_DARTS_Proximal", linestyle='-', linewidth='1.8')
# plt.legend(loc=4)
#
# # plt.title("Val_acc (%)")
# plt.xlabel("Epoch")
# plt.ylabel("Val_acc (%)")
# plt.savefig("search/darts_proximal_A.png")


# multiview = open(osp.join("search", "baseline", "log.txt")).readlines()
# milenas = open(osp.join("search", "milenas_baseline", "milenas_baseline" + ".log")).readlines()
#
# baseline_y = []
#
# milenas_baeline_only_A_x = []
# milenas_baeline_only_A_y = []
#
# for line in multiview:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         baseline_y.append(float(prec))
#
#
#
# for line in milenas:
#     if "valid_acc" in line:
#         prec = line.strip().split()[-1][:-1]
#         milenas_baeline_only_A_y.append(float(prec))
#
# multiview_x = np.linspace(0, len(baseline_y) * 1980 * 4, len(baseline_y))
# milenas_baeline_only_A_x = np.linspace(0, len(milenas_baeline_only_A_y) * 1980 * 2, len(milenas_baeline_only_A_y))
# plt.figure()
# plt.plot(multiview_x, baseline_y, label="VFL_DARTS", linewidth = '2')
# plt.plot(milenas_baeline_only_A_x, milenas_baeline_only_A_y, label="VFL_MiLeNAS", linewidth = '2')
# plt.legend(loc=4)
#
# # plt.title("Valid Accuracy vs. Communication Round", fontsize=14)
# plt.xlabel("Communication Round", fontsize=14)
# plt.ylabel("Val_Acc", fontsize=14)
# # plt.savefig(png_path)
# #
#
# # plt.title("Valid Accuracy")
# # plt.xlabel("Communication Round")
# # plt.ylabel("Accuracy")
# plt.savefig("search/vfl_communication.png")

def dp():
    multiview_32 = open(osp.join("search", "2_party_darts-20200916-120940", "log.txt")).readlines()
    multiview_32_dp = open(osp.join("search", "2_party_dp_out_1_grad_0-20200917-171651", "log.txt")).readlines()
    multiview_32_dp_3 = open(osp.join("search", "2_party_dp_out_3_grad_0-20200917-171641", "log.txt")).readlines()
    multiview_32_dp_10 = open(osp.join("search", "2_party_dp_out_10_grad_0-20200917-171711", "log.txt")).readlines()

    multiview_32_y = []

    multiview_32_dp_y = []

    multiview_32_dp_3_y = []
    multiview_32_dp_10_y = []

    for line in multiview_32:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            multiview_32_y.append(float(prec))

    for line in multiview_32_dp:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            multiview_32_dp_y.append(float(prec))

    for line in multiview_32_dp_3:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            multiview_32_dp_3_y.append(float(prec))

    for line in multiview_32_dp_10:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            multiview_32_dp_10_y.append(float(prec))

    multiview_32_x = np.linspace(0, len(multiview_32_y), len(multiview_32_y))
    multiview_32_dp_x = np.linspace(0, len(multiview_32_dp_y), len(multiview_32_dp_y))
    multiview_32_dp_3_x = np.linspace(0, len(multiview_32_dp_3_y), len(multiview_32_dp_3_y))
    multiview_32_dp_10_x = np.linspace(0, len(multiview_32_dp_10_y), len(multiview_32_dp_10_y))
    plt.figure()
    plt.plot(multiview_32_x, multiview_32_y, label="Baseline")
    plt.plot(multiview_32_dp_x, multiview_32_dp_y, label="DP-sigma-1.0")
    plt.plot(multiview_32_dp_3_x, multiview_32_dp_3_y, label="DP-sigma-3.0")
    plt.plot(multiview_32_dp_10_x, multiview_32_dp_10_y, label="DP-sigma-10.0")
    legend = plt.legend(loc=4)
    legend.get_title().set_fontsize(fontsize=40)

    # plt.title("Valid Accuracy vs. Epoch", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation Accuracy (%)", fontsize=14)
    # plt.savefig(png_path)

    plt.savefig("search/dp_epoch.png", bbox_inches='tight', dpi=320)

    best_acc = [multiview_32_y[-1], multiview_32_dp_y[-1], multiview_32_dp_3_y[-1], multiview_32_dp_10_y[-1]]
    dp_param = [0.0, 1.0, 3.0, 10.0]
    # func = interpolate.interp1d(dp_param, best_acc, kind='cubic')
    # x_new = np.arange(0,10,1)
    # x_new = np.arange(0, 4, 1)
    # y_new = func(dp_param)
    plt.figure()
    plt.plot(dp_param, best_acc, marker='o', ms=8, label="VFL with Noise")
    # plt.legend(loc=1, fontsize='x-large')
    plt.xlabel("Noise Standard Deviation", fontsize=14)
    plt.ylabel("Validation Accuracy (%)", fontsize=14)
    plt.grid(axis='x')
    plt.xticks(dp_param)
    plt.savefig("dp-sigma.png", bbox_inches='tight', dpi=320)
    # legend.get_title().set_fontsize(fontsize=14)


def selftrain():
    darts = open(osp.join("search", "single_train", "log.txt")).readlines()
    milenas = open(osp.join("search", "2_party_milenas-20200914-114331", "log.txt")).readlines()
    darts_moco = open(osp.join("search", "moco_v2", "log.txt")).readlines()
    milenas_moco = open(osp.join("search", "self-train-milenas-20201102-195522", "log.txt")).readlines()
    darts_moco_wo_alpha = open(
        osp.join("search", "self-train_without_finetune_alpha-20201103-160426", "log.txt")).readlines()
    milenas_moco_wo_alpha = open(
        osp.join("search", "self-train-milenas-without_fintune_alpha-20201103-141002", "log.txt")).readlines()

    darts_y = []
    milenas_y = []
    darts_moco_y = []
    milenas_moco_y = []
    darts_moco_wo_alpha_y = []
    milenas_moco_wo_alpha_y = []

    for line in darts:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            darts_y.append(float(prec))

    for line in milenas:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            milenas_y.append(float(prec))

    for line in darts_moco:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            darts_moco_y.append(float(prec))

    for line in milenas_moco:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            milenas_moco_y.append(float(prec))

    for line in darts_moco_wo_alpha:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            darts_moco_wo_alpha_y.append(float(prec))

    for line in milenas_moco_wo_alpha:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            milenas_moco_wo_alpha_y.append(float(prec))

    darts_x = np.linspace(0, len(darts_y) * 198 * 2, len(darts_y))
    milenas_x = np.linspace(0, len(milenas_y) * 198, len(milenas_y))
    darts_moco_x = np.linspace(0, len(darts_moco_y) * 198 * 2, len(darts_moco_y))
    milenas_moco_x = np.linspace(0, len(milenas_moco_y) * 198, len(milenas_moco_y))
    darts_moco_wo_alpha_x = np.linspace(0, len(darts_moco_wo_alpha_y) * 198, len(darts_moco_wo_alpha_y))
    milenas_moco_wo_alpha_x = np.linspace(0, len(milenas_moco_wo_alpha_y) * 198, len(milenas_moco_wo_alpha_y))

    plt.figure()
    plt.plot(darts_x, darts_y, label="VFL+DARTS")
    plt.plot(milenas_x, milenas_y, label="VFL+MiLeNAS")
    plt.plot(darts_moco_x, darts_moco_y, label="VFL+DARTS+MoCo")
    plt.plot(milenas_moco_x, milenas_moco_y, label="VFL+MiLeNAS+Moco")
    plt.plot(darts_moco_wo_alpha_x, darts_moco_wo_alpha_y, label="VFL+DARTS+MoCo_wo_alpha")
    plt.plot(milenas_moco_wo_alpha_x, milenas_moco_wo_alpha_y, label="VFL+MiLeNAS+MoCo_wo_alpha")

    legend = plt.legend(loc=4)
    legend.get_title().set_fontsize(fontsize=40)

    # plt.title("Valid Accuracy vs. Epoch", fontsize=14)
    plt.xlabel("Communication Round", fontsize=14)
    plt.ylabel("Validation Accuracy (%)", fontsize=14)
    # plt.savefig(png_path)

    plt.savefig("search/selftrain.png", bbox_inches='tight', dpi=320)


def dart_milenas():
    two_party_darts = open(osp.join("search", "2_party_darts-20200916-120940", "log.txt")).readlines()
    two_party_milenas = open(osp.join("search", "2_party_milenas-20200914-114331", "log.txt")).readlines()
    two_party_moco_darts_alpha = open(osp.join("search", "self-train_finetune_alpha_trainset-20201110-104738", "log.txt")).readlines()
    two_party_moco_darts_wo_alpha = open(
        osp.join("search", "self_train_without_finetune_alpha-20201110-195827", "log.txt")).readlines()
    two_party_moco_milenas_alpha = open(
        osp.join("search", "self_train_milenas_fintune_alpha-20201113-142514", "log.txt")).readlines()

    two_party_darts_y = []

    two_party_milenas_y = []
    two_party_moco_darts_alpha_y = []
    two_party_moco_darts_wo_alpha_y = []
    two_party_moco_milenas_alpha_y = []

    for line in two_party_darts:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            two_party_darts_y.append(float(prec))

    for line in two_party_milenas:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            two_party_milenas_y.append(float(prec))

    for line in two_party_moco_darts_alpha:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            two_party_moco_darts_alpha_y.append(float(prec))

    for line in two_party_moco_darts_wo_alpha:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            two_party_moco_darts_wo_alpha_y.append(float(prec))

    for line in two_party_moco_milenas_alpha:
        if "valid_acc" in line:
            prec = line.strip().split()[-1][:-1]
            two_party_moco_milenas_alpha_y.append(float(prec))

    two_party_darts_x = np.linspace(1, len(two_party_darts_y) * 198 * 2, len(two_party_darts_y))
    two_party_milenas_x = np.linspace(1, len(two_party_milenas_y) * 198, len(two_party_milenas_y))
    two_party_moco_darts_alpha_x = np.linspace(1, len(two_party_moco_darts_alpha_y) * 198 * 2, len(two_party_moco_darts_alpha_y))
    # two_party_moco_darts_wo_alpha_x = np.linspace(1, len(two_party_moco_darts_wo_alpha_y) * 198, len(two_party_moco_darts_wo_alpha_y))
    two_party_moco_milenas_alpha_x = np.linspace(1, len(two_party_moco_milenas_alpha_y) * 198, len(two_party_moco_milenas_alpha_y))
    plt.figure()
    ax = plt.gca()
    # ax.set_title("Test acc vs. num of parties")
    ax.set_xlabel("Communication Round", fontsize=14)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=14)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.figure()
    # plt.vlines(20, 0, 100)
    # plt.vlines(30, 0, 100)
    # plt.vlines(40, 0, 100)
    # plt.vlines(50, 0, 100)
    plt.plot(two_party_darts_x, two_party_darts_y, linestyle='--', linewidth=1.6, label="VFNAS$^{1}$")
    plt.plot(two_party_milenas_x, two_party_milenas_y, linestyle='--', linewidth=1.6, label="VFNAS$^{2}$")
    plt.plot(two_party_moco_darts_alpha_x, two_party_moco_darts_alpha_y, linestyle='--', linewidth=1.6, label="SS-VFNAS$^{1}$")
    # plt.plot(two_party_moco_darts_wo_alpha_x, two_party_moco_darts_wo_alpha_y, linestyle='--', linewidth=2.0, label="VFNAS-MiLeNAS")
    plt.plot(two_party_moco_milenas_alpha_x, two_party_moco_milenas_alpha_y, linestyle='--', linewidth=1.6, label="SS-VFNAS$^{2}$")

    # plt.scatter(two_party_darts_x[19], two_party_darts_y[19], c='dodgerblue')
    # plt.scatter(two_party_darts_x[29], two_party_darts_y[29], c='dodgerblue')
    # plt.scatter(two_party_darts_x[39], two_party_darts_y[39], c='dodgerblue')
    # plt.scatter(two_party_darts_x[49], two_party_darts_y[49], c='dodgerblue')

    plt.scatter(two_party_milenas_x[19], two_party_milenas_y[19], c='darkorange', marker='o', s=20)
    plt.scatter(two_party_milenas_x[29], two_party_milenas_y[29], c='darkorange', marker='s', s=20)
    plt.scatter(two_party_milenas_x[39], two_party_milenas_y[39], c='darkorange', marker='D', s=20)
    plt.scatter(two_party_milenas_x[49], two_party_milenas_y[49], c='darkorange', marker='<', s=20)

    plt.scatter(two_party_darts_x[19], two_party_darts_y[19], c='b', marker='o', s=20)
    plt.scatter(two_party_darts_x[29], two_party_darts_y[29], c='b', marker='s', s=20)
    plt.scatter(two_party_darts_x[39], two_party_darts_y[39], c='b', marker='D', s=20)
    plt.scatter(two_party_darts_x[49], two_party_darts_y[49], c='b', marker='<', s=20)
    #
    plt.scatter(two_party_moco_darts_alpha_x[19], two_party_moco_darts_alpha_y[19], c='g', marker='o', s=20)
    plt.scatter(two_party_moco_darts_alpha_x[29], two_party_moco_darts_alpha_y[29], c='g', marker='s', s=20)
    plt.scatter(two_party_moco_darts_alpha_x[39], two_party_moco_darts_alpha_y[39], c='g', marker='D', s=20)
    plt.scatter(two_party_moco_darts_alpha_x[49], two_party_moco_darts_alpha_y[49], c='g', marker='<', s=20)
    #
    plt.scatter(two_party_moco_milenas_alpha_x[19], two_party_moco_milenas_alpha_y[19], c='r', marker='o', s=30)
    plt.scatter(two_party_moco_milenas_alpha_x[29], two_party_moco_milenas_alpha_y[29], c='r', marker='s', s=30)
    plt.scatter(two_party_moco_milenas_alpha_x[39], two_party_moco_milenas_alpha_y[39], c='r', marker='D', s=30)
    plt.scatter(two_party_moco_milenas_alpha_x[49], two_party_moco_milenas_alpha_y[49], c='r', marker='<', s=30)
    #
    plt.grid(linestyle='-.')

    plt.legend(loc=4, fontsize=12)

    # # plt.title("Valid Accuracy vs. Epoch", fontsize=14)
    # # plt.xlabel("Epoch", fontsize=14)
    # plt.savefig(png_path)
    # plt.grid(axis='x')
    # plt.grid(axis='x')
    # dp_param = [two_party_milenas_x[19], two_party_milenas_x[29], two_party_milenas_x[39], two_party_milenas_x[49],
    #             two_party_darts_x[19], two_party_darts_x[29], two_party_darts_x[39], two_party_darts_x[49]]
    # plt.xticks(dp_param)
    dp_param = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
    # # plt.show()
    plt.savefig("search/dart_milenas.png", bbox_inches='tight', dpi=320)
    plt.figure()
    #
    ax = plt.gca()
    # ax.set_title("Test acc vs. num of parties")
    ax.set_xlabel("Communication Round", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    dp_param = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
    mile_best_acc = [82.125, 81.5, 82, 82.874997]
    darts_best_acc = [81.9375, 81.25, 81.8125, 81.9375]
    two_party_moco_darts_alpha_best_acc = [83.875, 83.625, 82.75, 82.875]
    two_party_moco_milenas_alpha_best_acc = [82.25, 82.5625, 83.06, 82.8125]
    # # func = interpolate.interp1d(dp_param, best_acc, kind='cubic')
    # # x_new = np.arange(0,10,1)
    # # x_new = np.arange(0, 4, 1)
    # # y_new = func(dp_param)
    # plt.figure()
    plt.plot(dp_param[4:], darts_best_acc, linestyle='--', linewidth=2.0, label="VFNAS$^{1}$")
    plt.plot(dp_param[:4], mile_best_acc, linestyle='--', linewidth=2.0, label="VFNAS$^{2}$")
    plt.plot(dp_param[4:], two_party_moco_darts_alpha_best_acc, linestyle='--', linewidth=2.0, label="Un-VFNAS$^{1}$")
    plt.plot(dp_param[:4], two_party_moco_milenas_alpha_best_acc, linestyle='--', linewidth=2.0, label="Un-VFNAS$^{2}$")

    plt.scatter(dp_param[0], mile_best_acc[0], c='r', marker='o', s=80)
    plt.scatter(dp_param[1], mile_best_acc[1], c='b', marker='o', s=80)
    plt.scatter(dp_param[2], mile_best_acc[2], c='g', marker='o', s=80)
    plt.scatter(dp_param[3], mile_best_acc[3], c='y', marker='o', s=80)

    plt.scatter(dp_param[4], darts_best_acc[0], c='r', marker='s', s=60)
    plt.scatter(dp_param[5], darts_best_acc[1], c='b', marker='s', s=60)
    plt.scatter(dp_param[6], darts_best_acc[2], c='g', marker='s', s=60)
    plt.scatter(dp_param[7], darts_best_acc[3], c='y', marker='s', s=60)

    plt.scatter(dp_param[4], two_party_moco_darts_alpha_best_acc[0], c='r', marker='s', s=60)
    plt.scatter(dp_param[5], two_party_moco_darts_alpha_best_acc[1], c='b', marker='s', s=60)
    plt.scatter(dp_param[6], two_party_moco_darts_alpha_best_acc[2], c='g', marker='s', s=60)
    plt.scatter(dp_param[7], two_party_moco_darts_alpha_best_acc[3], c='y', marker='s', s=60)

    plt.scatter(dp_param[0], two_party_moco_milenas_alpha_best_acc[0], c='r', marker='s', s=60)
    plt.scatter(dp_param[1], two_party_moco_milenas_alpha_best_acc[1], c='b', marker='s', s=60)
    plt.scatter(dp_param[2], two_party_moco_milenas_alpha_best_acc[2], c='g', marker='s', s=60)
    plt.scatter(dp_param[3], two_party_moco_milenas_alpha_best_acc[3], c='y', marker='s', s=60)


    # plt.grid(axis='x')
    # plt.xticks(dp_param)
    # plt.hlines(82, 3839, 9900, label="DARTS")
    plt.grid(linestyle='-.')
    plt.legend(loc=4, fontsize=12)
    plt.savefig("different_epoch.png", bbox_inches='tight', dpi=320)


# dp()
# commu_effe()
dart_milenas()
# selftrain()
