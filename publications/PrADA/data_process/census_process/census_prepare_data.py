from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from data_process import data_process_utils
from data_process.census_process.census_data_creation_config import census_data_creation
from data_process.census_process.census_degree_process_utils import consistentize_census9495_columns, \
    numericalize_census9495_data, standardize_census_data
from data_process.census_process.mapping_resource import cate_to_index_map, continuous_cols, categorical_cols, \
    target_col_name


# follow link provides description on columns of Census Income Dataset:
# https://docs.1010data.com/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html

def get_timestamp():
    return int(datetime.utcnow().timestamp())


CENSUS_COLUMNS = ["age", "class_worker", "det_ind_code", "det_occ_code", "education",
                  "wage_per_hour", "hs_college", "marital_stat", "major_ind_code", "major_occ_code",
                  "race", "hisp_origin", "gender", "union_member", "unemp_reason", "full_or_part_emp",
                  "capital_gain", "capital_loss", "stock_dividends", "tax_filer_stat",
                  "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ", "instance_weight",
                  "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
                  "num_emp", "fam_under_18", "country_father", "country_mother", "country_self",
                  "citizenship", "own_or_self", "vet_question", "vet_benefits", "weeks_worked",
                  "year", "income_label"]

RERANGED_CENSUS_COLUMNS_NEW = ["age", "gender_index", "age_index", "class_worker", "det_ind_code", "det_occ_code",
                               "education",
                               "education_year", "wage_per_hour", "hs_college", "marital_stat", "major_ind_code",
                               "major_occ_code", "race", "hisp_origin", "gender", "union_member", "unemp_reason",
                               "full_or_part_emp", "capital_gain", "capital_loss", "stock_dividends", "tax_filer_stat",
                               "region_prev_res", "state_prev_res", "det_hh_fam_stat", "det_hh_summ", "instance_weight",
                               "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same", "mig_prev_sunbelt",
                               "num_emp", "fam_under_18", "country_father", "country_mother", "country_self",
                               "citizenship", "own_or_self", "vet_question", "vet_benefits", "weeks_worked",
                               "year", "income_label"]


def process(data_path, to_dir=None, train=True):
    census = pd.read_csv(data_path, names=CENSUS_COLUMNS, skipinitialspace=True)
    print("[INFO] load {} data".format("train" if train else "test"))
    print("[INFO] load data with shape:", census.shape)

    appendix = "_train" if train else "_test"
    extension = ".csv"
    appendix = appendix + extension

    print("[INFO] consistentize original data")
    c_census = consistentize_census9495_columns(census)
    c_census.to_csv(to_dir + 'consistentized_census9495' + appendix, header=True, index=False)

    print("[INFO] numericalize data")
    p_census = numericalize_census9495_data(c_census, cate_to_index_map)
    return p_census


def compute_instance_prob(data_frame):
    weight_sum = data_frame["instance_weight"].sum()
    data_frame["instance_weight"] = data_frame["instance_weight"] / weight_sum


def create_file_appendix(train):
    appendix = "_train" if train else "_valid"
    extension = ".csv"
    return appendix + extension


def create_degree_src_tgt_data(p_census,
                               from_dir,
                               to_dir,
                               data_tag,
                               pos_ratio,
                               num_all,
                               train=True,
                               grad_train_scaler=None,
                               undergrad_train_scaler=None,
                               grad_census_test_values=None,
                               save_intermediate_tables=False):
    appendix = create_file_appendix(train)
    print("====================== create_degree_source_target_data for {} data ======================"
          .format("train" if train else "valid"))

    # form source and target domain data
    doctorate_census = p_census[p_census['education'] == 11]
    master_census = p_census[(p_census['education'] == 9) | (p_census['education'] == 10)]
    undergrad_census = p_census[
        (p_census['education'] != 9) & (p_census['education'] != 10) & (p_census['education'] != 11)]
    columns = continuous_cols + categorical_cols + ['instance_weight', target_col_name]
    doctorate_census = doctorate_census[columns]
    master_census = master_census[columns]
    undergrad_census = undergrad_census[columns]
    print("[INFO] doctorate_census shape", doctorate_census.shape)
    print("[INFO] master_census shape", master_census.shape)
    print("[INFO] undergrad_census shape", undergrad_census.shape)

    if save_intermediate_tables:
        doctorate_census.to_csv(to_dir + 'doctorate_census9495' + appendix, header=True, index=False)
        master_census.to_csv(to_dir + 'master_census9495' + appendix, header=True, index=False)
        undergrad_census.to_csv(to_dir + 'undergrad_census9495' + appendix, header=True, index=False)

        doctorate_census = pd.read_csv(from_dir + 'doctorate_census9495' + appendix, skipinitialspace=True)
        master_census = pd.read_csv(from_dir + 'master_census9495' + appendix, skipinitialspace=True)
        undergrad_census = pd.read_csv(from_dir + 'undergrad_census9495' + appendix, skipinitialspace=True)

    doctorate_census_values = doctorate_census[columns].values
    master_census_values = master_census[columns].values
    undergrad_census_values = undergrad_census[columns].values

    # doctor and master form the source domain
    grad_census_values = np.concatenate([doctorate_census_values, master_census_values], axis=0)
    grad_census_values = shuffle(grad_census_values)
    grad_census_df_for_da = pd.DataFrame(data=grad_census_values, columns=columns)

    # undergraduate form the target domain
    undergrad_census_values = shuffle(undergrad_census_values)
    undergrad_census_df = pd.DataFrame(data=undergrad_census_values, columns=columns)

    _, grad_train_scaler = standardize_census_data(grad_census_df_for_da, continuous_cols, grad_train_scaler)
    _, udgrad_train_scaler = standardize_census_data(undergrad_census_df, continuous_cols, undergrad_train_scaler)

    grad_census_df_1 = grad_census_df_for_da[grad_census_df_for_da[target_col_name] == 1]
    grad_census_df_0 = grad_census_df_for_da[grad_census_df_for_da[target_col_name] == 0]

    undergrad_census_df_1 = undergrad_census_df[undergrad_census_df[target_col_name] == 1]
    undergrad_census_df_0 = undergrad_census_df[undergrad_census_df[target_col_name] == 0]

    print("[INFO] (orig) (target) grad_census_df_1 shape:", grad_census_df_1.shape)
    print("[INFO] (orig) (target) grad_census_df_0 shape:", grad_census_df_0.shape)
    print("[INFO] (orig) (source) undergrad_census_df_1 shape:", undergrad_census_df_1.shape)
    print("[INFO] (orig) (source) undergrad_census_df_0 shape:", undergrad_census_df_0.shape)

    grad_census_for_test = None
    test_pos_ratio = 0.5
    if train:
        num_pos = int(num_all * pos_ratio)
        num_neg = int(num_all * (1 - pos_ratio))

        print(f"[INFO] train num_pos:{num_pos}")
        print(f"[INFO] train num_neg:{num_neg}")

        # get labeled target data for supervised training
        grad_census_values_1 = grad_census_df_1.values[0:num_pos]
        grad_census_values_0 = grad_census_df_0.values[0:num_neg]
        grad_census_values_for_supervise = shuffle(np.concatenate((grad_census_values_1, grad_census_values_0), axis=0))
        print(f"[INFO] grad train positive samples range:[0:{num_pos}].")
        print(f"[INFO] grad train negative samples range:[0:{num_neg}].")
        print(f"[INFO] grad train all samples shape:{grad_census_values_for_supervise.shape}.")

        num_pos_for_test = int((grad_census_df_0.shape[0] - num_all) * test_pos_ratio)
        grad_census_test_values_1 = grad_census_df_1.values[num_pos:num_pos + num_pos_for_test]
        grad_census_test_values_0 = grad_census_df_0.values[num_all:]
        print(f"[INFO] => grad left_data for test # of positive samples:{num_pos_for_test}")
        print(f"[INFO] => grad left-data for test pos samples range:[{num_pos}:{num_pos + num_pos_for_test}].")
        print(f"[INFO] => grad left-data for test pos samples shape:{grad_census_test_values_1.shape}")
        print(f"[INFO] => grad left-data for test neg samples range:[{num_all}:-1].")
        print(f"[INFO] => grad left-data for test neg samples shape:{grad_census_test_values_0.shape}")

        grad_census_for_test = np.concatenate([grad_census_test_values_1, grad_census_test_values_0], axis=0)
        print(f"[INFO] => grad left-data for test shape: {grad_census_for_test.shape}")

    else:
        # num_pos = int((grad_census_df_0.shape[0] + grad_census_df_0.shape[1]) * test_pos_ratio)
        # grad_census_values_1 = grad_census_df_1.values[:num_pos]
        grad_census_values_1 = grad_census_df_1.values
        grad_census_values_0 = grad_census_df_0.values
        grad_census_values_for_supervise = shuffle(
            np.concatenate((grad_census_values_1, grad_census_values_0, grad_census_test_values), axis=0))
        print(f"[INFO] grad test pos samples shape:{grad_census_values_1.shape}.")
        print(f"[INFO] grad test neg samples shape:{grad_census_values_0.shape}.")
        print(f"[INFO] grad left-data for test samples shape:{grad_census_test_values.shape}.")
        print(f"[INFO] grad test all samples shape: {grad_census_values_for_supervise.shape}")

    # print("grad_census_values_1 shape:", grad_census_values_1.shape)
    # print("grad_census_values_0 shape:", grad_census_values_0.shape)

    # grad_census_values_for_supervise = shuffle(np.concatenate((grad_census_values_1, grad_census_values_0), axis=0))
    grad_census_df_for_ft = pd.DataFrame(data=grad_census_values_for_supervise, columns=columns)
    print("[INFO] (final) grad_census_df_for_ft (supervised) shape:", grad_census_df_for_ft.shape)
    print("[INFO]         grad_census_df_for_ft (supervised) pos:",
          grad_census_df_for_ft[grad_census_df_for_ft[target_col_name] == 1].shape)
    print("[INFO]         grad_census_df_for_ft (supervised) neg:",
          grad_census_df_for_ft[grad_census_df_for_ft[target_col_name] == 0].shape)

    # save data
    if train:
        grad_ft_file_full_path = from_dir + 'grad_census9495_ft_' + str(data_tag) + appendix
        grad_census_df_for_ft.to_csv(grad_ft_file_full_path, header=True, index=False)
        print(f"[INFO] ==> saved grad ft data to: {grad_ft_file_full_path}")

        print("[INFO] (final) grad_census_df_for_ad shape:", grad_census_df_for_da.shape)
        print("[INFO]         grad_census_df_for_ad pos:",
              grad_census_df_for_da[grad_census_df_for_da[target_col_name] == 1].shape)
        print("[INFO]         grad_census_df_for_ad neg:",
              grad_census_df_for_da[grad_census_df_for_da[target_col_name] == 0].shape)
        grad_da_file_full_path = from_dir + 'grad_census9495_ad_' + str(data_tag) + appendix
        grad_census_df_for_da.to_csv(grad_da_file_full_path, header=True, index=False)
        print(f"[INFO] ==>  saved grad ad data to: {grad_da_file_full_path}")

    else:
        # test
        half_num = int(grad_census_df_for_ft.shape[0] / 2)
        grad_census_df_for_ft_valid = grad_census_df_for_ft[:half_num]
        grad_census_df_for_ft_test = grad_census_df_for_ft[half_num:]

        print(f"[INFO] (final) grad_census_df_for_ft_valid shape:{grad_census_df_for_ft_valid.shape}")
        print(f"[INFO]      => grad_census_df_for_ft_valid shape range:[0:{half_num}].")
        print("[INFO]         grad_census_df_for_ft_valid pos:",
              grad_census_df_for_ft_valid[grad_census_df_for_ft_valid[target_col_name] == 1].shape)
        print("[INFO]         grad_census_df_for_ft_valid neg:",
              grad_census_df_for_ft_valid[grad_census_df_for_ft_valid[target_col_name] == 0].shape)
        grad_ft_file_full_path = from_dir + 'grad_census9495_ft_' + str(data_tag) + "_valid.csv"
        grad_census_df_for_ft_valid.to_csv(grad_ft_file_full_path, header=True, index=False)
        print(f"[INFO] ==>  saved grad ft valid data to: {grad_ft_file_full_path}")

        print(f"[INFO] (final) grad_census_df_for_ft_test shape:{grad_census_df_for_ft_test.shape}")
        print(f"[INFO]      => grad_census_df_for_ft_test range:[{half_num}:].")
        print("[INFO]         grad_census_df_for_ft_test pos:",
              grad_census_df_for_ft_test[grad_census_df_for_ft_test[target_col_name] == 1].shape)
        print("[INFO]         grad_census_df_for_ft_valid neg:",
              grad_census_df_for_ft_test[grad_census_df_for_ft_test[target_col_name] == 0].shape)
        grad_ft_file_full_path = from_dir + 'grad_census9495_ft_' + str(data_tag) + "_test.csv"
        grad_census_df_for_ft_test.to_csv(grad_ft_file_full_path, header=True, index=False)
        print(f"[INFO] ==>  saved grad ft test data to: {grad_ft_file_full_path}")

    undergrad_pos_num = undergrad_census_df_1.shape[0]
    undergrad_census_values_all = shuffle(
        np.concatenate((undergrad_census_df_1.values, undergrad_census_df_0[:undergrad_pos_num * 9].values), axis=0))
    undergrad_census_df_all = pd.DataFrame(data=undergrad_census_values_all, columns=columns)

    print("[INFO] (final) undergrad_census_df_all shape:", undergrad_census_df_all.shape)
    print("[INFO]         undergrad_census_df_all pos:",
          undergrad_census_df_all[undergrad_census_df_all[target_col_name] == 1].shape)
    print("[INFO]         undergrad_census_df_all neg:",
          undergrad_census_df_all[undergrad_census_df_all[target_col_name] == 0].shape)
    undergrad_file_full_path = from_dir + 'undergrad_census9495_ad_' + str(data_tag) + appendix
    undergrad_census_df_all.to_csv(undergrad_file_full_path, header=True, index=False)
    print(f"[INFO] ==> saved undergrad ad data to: {undergrad_file_full_path}")

    return grad_train_scaler, udgrad_train_scaler, grad_census_for_test


def combine_src_tgt_data(from_dir, to_dir, data_tag):
    print(f"========================= combine census source and target data ============================ ")

    source_train_file_name = from_dir + f'undergrad_census9495_ad_{data_tag}_train.csv'
    target_train_file_name = from_dir + f'grad_census9495_ft_{data_tag}_train.csv'

    df_src_data = pd.read_csv(source_train_file_name, skipinitialspace=True)
    df_tgt_data = pd.read_csv(target_train_file_name, skipinitialspace=True)
    print("[INFO] df_src_data shape:", df_src_data.shape)
    print("[INFO] df_tgt_data shape:", df_tgt_data.shape)

    df_data = data_process_utils.combine_src_tgt_data(df_src_data, df_tgt_data)
    print("[INFO] df_src_tgt_data shape:", df_data.shape)

    file_full_name = "{}/degree_src_tgt_census9495_{}_train.csv".format(to_dir, data_tag)
    data_process_utils.save_df_data(df_data, file_full_name)


if __name__ == "__main__":
    data_dir = census_data_creation['original_data_dir']
    output_data_dir = census_data_creation['processed_data_dir']

    data_tag = census_data_creation['data_tag']
    pos_ratio = census_data_creation['positive_sample_ratio']
    num_all = census_data_creation['number_target_samples']

    print("[INFO] ------ process data ------")
    train_data_path = data_dir + census_data_creation['train_data_file_name']
    test_data_path = data_dir + census_data_creation['test_data_file_name']
    train_df = process(train_data_path, to_dir=output_data_dir, train=True)
    test_df = process(test_data_path, to_dir=output_data_dir, train=False)

    grad_train_scaler, udgrad_train_scaler, grad_census_for_test = create_degree_src_tgt_data(train_df,
                                                                                              from_dir=output_data_dir,
                                                                                              to_dir=output_data_dir,
                                                                                              train=True,
                                                                                              pos_ratio=pos_ratio,
                                                                                              num_all=num_all,
                                                                                              data_tag=data_tag)
    create_degree_src_tgt_data(test_df,
                               from_dir=output_data_dir,
                               to_dir=output_data_dir,
                               train=False,
                               pos_ratio=pos_ratio,
                               grad_train_scaler=grad_train_scaler,
                               undergrad_train_scaler=udgrad_train_scaler,
                               grad_census_test_values=grad_census_for_test,
                               data_tag=data_tag,
                               num_all=num_all)

    # NOTE: input dir and output dir are the same for src_tgt_data
    combine_src_tgt_data(from_dir=output_data_dir, to_dir=output_data_dir, data_tag=data_tag)
