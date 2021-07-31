from collections import OrderedDict

from experiments.ppd_loan.train_config import feature_extractor_architecture_list, data_tag, \
    pre_train_hyperparameters, data_hyperparameters
from experiments.ppd_loan.meta_data import column_name_list, group_ind_list, group_info, embedding_shape_map
from experiments.ppd_loan.train_ppd_utils import pretrain_ppd
from models.classifier import CensusFeatureAggregator
from models.dann_models import create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_fg_dann_global_model


def record_domain_data(domain_data_dict, domain_data, domain_col_list, is_categorical):
    if is_categorical:
        emb_dict = OrderedDict()
        for col_index, col_name in enumerate(domain_col_list):
            emb_dict[col_name] = domain_data[:, col_index]
        domain_data_dict["embeddings"] = emb_dict

    else:
        domain_data_dict["non_embedding"] = {"tabular_data": domain_data}


def parse_domain_data(data, column_name_list, df_group_ind_list, df_group_info_list):
    wide_feat_list = []
    domain_list = []
    for group_ind, group_info in zip(df_group_ind_list, df_group_info_list):
        start_index = group_info[0]
        # 2 stands for wide features (i.e., features for active parties)
        if group_ind == 2:
            length, _ = group_info[1]
            wide_feat_list.append(data[:, start_index:start_index + length])
        # 1 stands for feature group (i.e., feature for passive party)
        elif group_ind == 1:
            new_domain = dict()
            new_domain["embeddings"] = None
            new_domain["non_embedding"] = None
            tuple_num = len(group_info)
            for i in range(1, tuple_num):
                length, is_cat = group_info[i]
                domain_data = data[:, start_index:start_index + length]
                domain_col_list = column_name_list[start_index:start_index + length]
                record_domain_data(new_domain, domain_data, domain_col_list, is_cat)
                start_index = start_index + length
            domain_list.append(new_domain)
    return wide_feat_list, domain_list


def partition_data(data):
    """
    partition data into party C and party A(B). Data in party C is partitioned into group
    """

    wide_feat_list, domain_list = parse_domain_data(data,
                                                    column_name_list,
                                                    group_ind_list,
                                                    group_info)
    return wide_feat_list, domain_list


def create_model_group(extractor_input_dim):
    """
    create a group of models, namely feature extractor, aggregator and discriminator, for each feature group
    """

    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    classifier = CensusFeatureAggregator(input_dim=extractor_input_dim[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
    return extractor, classifier, discriminator


def create_embedding_dict():
    """
    create embedding dictionary for categorical features/columns
    """

    embedding_meta_dict = dict()
    for feat_name, emb_shape in embedding_shape_map.items():
        embedding_meta_dict[feat_name] = emb_shape
    print(f"embedding_meta_dict: \n {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def create_fg_pdd_global_model(pos_class_weight=1.0, num_wide_feature=6, using_interaction=False):
    embedding_dict = create_embedding_dict()

    num_wide_feature = num_wide_feature
    using_feature_group = True
    global_model = wire_fg_dann_global_model(embedding_dict=embedding_dict,
                                             feat_extr_archit_list=feature_extractor_architecture_list,
                                             intr_feat_extr_archit_list=None,
                                             num_wide_feature=num_wide_feature,
                                             using_feature_group=using_feature_group,
                                             using_interaction=using_interaction,
                                             partition_data_fn=partition_data,
                                             create_model_group_fn=create_model_group,
                                             pos_class_weight=pos_class_weight)

    return global_model


if __name__ == "__main__":
    ppd_dann_root_dir = data_hyperparameters['ppd_fg_pretrained_model_dir']
    using_interaction = pre_train_hyperparameters['using_interaction']
    pos_class_weight = pre_train_hyperparameters['pos_class_weight']
    init_model = create_fg_pdd_global_model(using_interaction=using_interaction,
                                            pos_class_weight=pos_class_weight)

    task_id = pretrain_ppd(data_tag,
                           ppd_dann_root_dir,
                           pre_train_hyperparameters,
                           data_hyperparameters,
                           model=init_model,
                           apply_feature_group=True)
    print(f"[INFO] adaptation task id:{task_id}")
