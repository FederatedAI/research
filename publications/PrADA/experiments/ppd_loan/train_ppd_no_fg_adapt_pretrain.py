from collections import OrderedDict

import torch

from experiments.ppd_loan.train_config import data_tag, \
    pre_train_hyperparameters, data_hyperparameters, no_fg_feature_extractor_architecture
from experiments.ppd_loan.meta_data import column_name_list, group_ind_list, group_info
from experiments.ppd_loan.train_ppd_fg_adapt_pretrain import parse_domain_data, create_embedding_dict
from experiments.ppd_loan.train_ppd_utils import pretrain_ppd
from models.classifier import GlobalClassifier, CensusFeatureAggregator
from models.dann_models import GlobalModel, RegionalModel
from models.discriminator import LendingRegionDiscriminator
from models.feature_extractor import CensusRegionFeatureExtractorDense


def store_domain_data(domain_data_dict, domain_data, domain_col_list, is_cat):
    if is_cat:
        emb_dict = OrderedDict()
        for col_index, col_name in enumerate(domain_col_list):
            emb_dict[col_name] = domain_data[:, col_index]
        domain_data_dict["embeddings"] = emb_dict

    else:
        domain_data_dict["non_embedding"] = {"tabular_data": domain_data}


def aggregate_domains(domain_list):
    agg_domain = dict({'embeddings': None, 'non_embedding': dict()})
    agg_embed_dict = dict()
    non_embed_list = []
    for domain in domain_list:
        embed_dict = domain['embeddings']
        if embed_dict:
            agg_embed_dict.update(embed_dict)
        non_embed = domain['non_embedding']
        if non_embed:
            non_embed_list.append(non_embed['tabular_data'])

    agg_domain['embeddings'] = agg_embed_dict
    agg_domain['non_embedding']['tabular_data'] = torch.cat(non_embed_list, dim=1)
    return agg_domain


def partition_data(data):
    wide_feat_list, domain_list = parse_domain_data(data,
                                                    column_name_list,
                                                    group_ind_list,
                                                    group_info)
    agg_domain = aggregate_domains(domain_list)
    return wide_feat_list, [agg_domain]


def create_region_model(extractor_input_dims_list, aggregation_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dims_list[-1], output_dim=aggregation_dim)
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=aggregator,
                         discriminator=discriminator)


def create_region_model_list(feature_extractor_arch_list, aggregation_dim):
    model_list = list()
    for feature_extractor_arch in feature_extractor_arch_list:
        model_list.append(create_region_model(feature_extractor_arch, aggregation_dim))
    return model_list


def create_no_fg_pdd_global_model(aggregation_dim=5, num_wide_feature=6, pos_class_weight=1.0):
    embedding_dict = create_embedding_dict()

    feature_extractor_architecture = no_fg_feature_extractor_architecture
    print(f"[INFO] feature_extractor_architecture list:{[feature_extractor_architecture]}")

    region_wrapper_list = create_region_model_list([feature_extractor_architecture], aggregation_dim)
    global_input_dim = aggregation_dim + num_wide_feature
    print(f"[INFO] global_input_dim: {global_input_dim}")
    source_classifier = GlobalClassifier(input_dim=global_input_dim)
    wrapper = GlobalModel(source_classifier=source_classifier,
                          regional_model_list=region_wrapper_list,
                          embedding_dict=embedding_dict,
                          partition_data_fn=partition_data,
                          pos_class_weight=pos_class_weight,
                          loss_name="BCE")
    return wrapper


if __name__ == "__main__":
    pretrained_model_dir = data_hyperparameters["ppd_no-fg_pretrained_model_dir"]
    pos_class_weight = pre_train_hyperparameters['pos_class_weight']
    init_model = create_no_fg_pdd_global_model(pos_class_weight=pos_class_weight)
    task_id = pretrain_ppd(data_tag,
                           pretrained_model_dir,
                           pre_train_hyperparameters,
                           data_hyperparameters,
                           model=init_model,
                           apply_feature_group=False)

