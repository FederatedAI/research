import numpy as np

from models.classifier import GlobalClassifier
from models.dann_models import GlobalModel, RegionalModel
from models.discriminator import GlobalDiscriminator
from models.interaction_models import InteractionModel, BiFeatureInteractionComputer
from utils import compute_parameter_size


def create_feature_interaction_model(feature_extractor_architecture_list,
                                     intr_feature_extractor_architecture_list,
                                     create_model_group_fn):
    """
    create interaction model that is responsible for interations among feature groups.
    """

    feat_intr_computer = BiFeatureInteractionComputer()
    if intr_feature_extractor_architecture_list is None:
        intr_feat_extractor_arch_list = list()
        for i in range(len(feature_extractor_architecture_list) - 1):
            for j in range(i + 1, len(feature_extractor_architecture_list)):
                print(i, "---", j)
                intr_feat_extractor_arch_list.append(
                    np.array(feature_extractor_architecture_list[i]) + np.array(feature_extractor_architecture_list[j]))
    else:
        intr_feat_extractor_arch_list = intr_feature_extractor_architecture_list
    print(f"intr_feat_extractor_arch_list:{intr_feat_extractor_arch_list}")
    print(f"# of parameters:{compute_parameter_size(intr_feat_extractor_arch_list)}")
    extractor_list, classifier_list, discriminator_list = create_model_group_list(intr_feat_extractor_arch_list,
                                                                                  create_model_group_fn)
    return InteractionModel(extractor_list=extractor_list,
                            aggregator_list=classifier_list,
                            discriminator_list=discriminator_list,
                            interactive_feature_computer=feat_intr_computer)


def create_model_group_list(input_dims_list, create_model_group_fn):
    """
    create models for each interactive feature group
    """
    extractor_list = list()
    classifier_list = list()
    discriminator_list = list()
    for input_dim in input_dims_list:
        models = create_model_group_fn(input_dim)
        extractor_list.append(models[0])
        classifier_list.append(models[1])
        discriminator_list.append(models[2])
    return extractor_list, classifier_list, discriminator_list


def create_region_model(extractor_input_dim, create_model_group_fn):
    """
    create models, namely feature extractor, aggregator and discriminator, for a region representing a feature group.
    """
    extractor, aggregator, discriminator = create_model_group_fn(extractor_input_dim)
    return RegionalModel(extractor=extractor, aggregator=aggregator, discriminator=discriminator)


def create_region_model_list(input_dims_list, create_model_group_fn):
    """
    create models for all regions that each represents a feature group.
    """
    wrapper_list = list()
    for input_dims in input_dims_list:
        wrapper_list.append(create_region_model(input_dims, create_model_group_fn))
    return wrapper_list


def wire_fg_dann_global_model(embedding_dict,
                              feat_extr_archit_list,
                              intr_feat_extr_archit_list,
                              num_wide_feature,
                              using_feature_group,
                              using_interaction,
                              partition_data_fn,
                              create_model_group_fn,
                              pos_class_weight=1.0):
    """
    wire up all models together as a single model for end-to-end training

    parameters:
    ----------
    embedding_dict - the embedding dictionary,
    feat_extr_archit_list - the neural network architecture for all feature groups (neural network
    has only dense layers),
    intr_feat_extr_archit_list - the neural network architecture for interactive feature groups
    (neural network has only dense layers),
    num_wide_feature - the number of feature used in party A or party B,
    using_feature_group - whether apply feature group,
    using_interaction -  whether apply interactions among feature groups,
    partition_data_fn - the data partition function,
    create_model_group_fn - the create model group function,
    pos_class_weight - weights for positive classes
    """

    if using_feature_group:
        print(f"[INFO] feature_extractor_architecture_list:{feat_extr_archit_list}, "
              f"len:{len(feat_extr_archit_list)}")
        print(f"# of parameters:{compute_parameter_size(feat_extr_archit_list)}")
        region_model_list = create_region_model_list(feat_extr_archit_list, create_model_group_fn)
    else:
        region_model_list = list()
    print(f"[INFO] region_model_list len:{len(region_model_list)}")

    interaction_model = None
    interactive_group_num = 0
    if using_interaction:
        interaction_model = create_feature_interaction_model(feat_extr_archit_list,
                                                             intr_feat_extr_archit_list,
                                                             create_model_group_fn)
        interactive_group_num = interaction_model.get_num_feature_groups()

    global_discriminator_dim = len(region_model_list) + interactive_group_num
    global_input_dim = num_wide_feature + len(region_model_list) + interactive_group_num

    print(f"[INFO] global_input_dim length:{global_input_dim}")
    print(f"[INFO] global_discriminator_dim length:{global_discriminator_dim}")

    source_classifier = GlobalClassifier(input_dim=global_input_dim)
    global_discriminator = GlobalDiscriminator(input_dim=global_discriminator_dim)
    global_model = GlobalModel(source_classifier=source_classifier,
                               regional_model_list=region_model_list,
                               embedding_dict=embedding_dict,
                               partition_data_fn=partition_data_fn,
                               feature_interaction_model=interaction_model,
                               pos_class_weight=pos_class_weight,
                               loss_name="BCE",
                               discriminator=global_discriminator)
    return global_model
