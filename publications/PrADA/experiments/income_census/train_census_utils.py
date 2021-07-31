from datasets.census_dataloader import get_income_census_dataloaders
from models.experiment_adaptation_pretrain_learner import FederatedDAANLearner
from models.experiment_finetune_target_learner import FederatedTargetLearner
from utils import get_timestamp, get_current_date, create_id_from_hyperparameters, test_classifier


def pretrain_census(data_tag,
                    census_dann_root_dir,
                    learner_hyperparameters,
                    data_hyperparameters,
                    model,
                    apply_feature_group=True):

    # hyper-parameters
    using_interaction = learner_hyperparameters['using_interaction']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    epoch_patience = learner_hyperparameters['epoch_patience']
    max_epochs = learner_hyperparameters['max_epochs']
    valid_metric = learner_hyperparameters['valid_metric']

    date = get_current_date()
    timestamp = get_timestamp()

    optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}}

    # create pre-train task id
    hyperparameter_dict = {"lr": lr, "bs": batch_size, "me": max_epochs, "ts": timestamp}
    if apply_feature_group:
        using_intr_tag = "intr" + str(True) if using_interaction else "intr" + str(False)
        task_id = date + "_census_fg_adapt_" + data_tag + "_" + using_intr_tag + "_" + create_id_from_hyperparameters(
            hyperparameter_dict)
    else:
        task_id = date + "_census_no_fg_adapt_" + data_tag + "_" + create_id_from_hyperparameters(hyperparameter_dict)
    print("[INFO] perform task:{0}".format(task_id))

    # load data
    source_train_file_name = data_hyperparameters['source_ad_train_file_name']
    source_valid_file_name = data_hyperparameters['source_ad_valid_file_name']
    target_train_file_name = data_hyperparameters['target_ad_train_file_name']
    target_valid_file_name = data_hyperparameters['target_ft_valid_file_name']

    print(f"[INFO] load source train from: {source_train_file_name}.")
    print(f"[INFO] load source valid from: {source_valid_file_name}.")
    print(f"[INFO] load target train from: {target_train_file_name}.")
    print(f"[INFO] load target valid from: {target_valid_file_name}.")

    print("[INFO] Load train data.")
    source_da_census_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=source_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_da_census_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load valid data.")
    source_census_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=source_valid_file_name, batch_size=batch_size * 4, split_ratio=1.0)
    target_census_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_valid_file_name, batch_size=batch_size * 4, split_ratio=1.0)

    plat = FederatedDAANLearner(model=model,
                                source_da_train_loader=source_da_census_train_loader,
                                source_val_loader=source_census_valid_loader,
                                target_da_train_loader=target_da_census_train_loader,
                                target_val_loader=target_census_valid_loader,
                                max_epochs=max_epochs,
                                epoch_patience=epoch_patience)
    plat.set_model_save_info(census_dann_root_dir)

    plat.train_dann(epochs=200,
                    task_id=task_id,
                    metric=valid_metric,
                    optimizer_param_dict=optimizer_param_dict)
    return task_id


def finetune_census(pretrain_task_id,
                    census_pretain_model_root_dir,
                    census_finetune_target_root_dir,
                    learner_hyperparameters,
                    data_hyperparameters,
                    model):

    # hyper-parameters
    load_global_classifier = learner_hyperparameters['load_global_classifier']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    valid_metric = learner_hyperparameters['valid_metric']

    date = get_current_date()
    timestamp = get_timestamp()

    # create fine-tune task id
    hyperparameter_dict = {"lr": lr, "bs": batch_size, "ts": timestamp}
    appendix = create_id_from_hyperparameters(hyperparameter_dict)
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    target_task_id = pretrain_task_id + "@target_" + date + "_" + glr + "_" + appendix
    print("[INFO] perform task:{0}".format(target_task_id))

    # load pre-trained model
    model.load_model(root=census_pretain_model_root_dir,
                     task_id=pretrain_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    target_ft_train_file_name = data_hyperparameters['target_ft_train_file_name']
    target_ft_valid_file_name = data_hyperparameters['target_ft_valid_file_name']
    target_ft_test_file_name = data_hyperparameters['target_ft_test_file_name']
    print(f"[INFO] load target ft train data from {target_ft_train_file_name}.")
    print(f"[INFO] load target ft valid data from {target_ft_valid_file_name}.")
    print(f"[INFO] load target ft test data from {target_ft_test_file_name}.")

    print("[INFO] Load train data")
    target_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_ft_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_ft_valid_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_test_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_ft_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info(census_finetune_target_root_dir)

    plat_target.train_target_with_alternating(global_epochs=500,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              metric=valid_metric,
                                              momentum=momentum,
                                              weight_decay=weight_decay)

    # load best model
    model.load_model(root=census_finetune_target_root_dir,
                     task_id=target_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classifier(model, target_test_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
    return target_task_id


def train_no_adaptation(data_tag,
                        census_no_ad_root_dir,
                        learner_hyperparameters,
                        data_hyperparameters,
                        model):

    # hyper-parameters
    apply_feature_group = learner_hyperparameters['apply_feature_group']
    train_data_tag = learner_hyperparameters['train_data_tag']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    epoch_patience = learner_hyperparameters['epoch_patience']
    max_epochs = learner_hyperparameters['max_epochs']
    valid_metric = learner_hyperparameters['valid_metric']

    src_tgt_train_file_name = data_hyperparameters['src_tgt_train_file_name']
    source_valid_file_name = data_hyperparameters['source_ad_valid_file_name']

    target_ft_train_file_name = data_hyperparameters['target_ft_train_file_name']
    target_ft_valid_file_name = data_hyperparameters['target_ft_valid_file_name']
    target_ft_test_file_name = data_hyperparameters['target_ft_test_file_name']

    data_file_name_dict = {"tgt": target_ft_train_file_name,
                           "all": src_tgt_train_file_name}

    # print(f"[INFO] load source train from: {source_train_file_name}.")
    print(f"[INFO] load source valid from: {source_valid_file_name}.")
    print(f"[INFO] load target train from: {target_ft_train_file_name}.")
    print(f"[INFO] load target valid from: {target_ft_valid_file_name}.")
    print(f"[INFO] load target test from: {target_ft_test_file_name}.")
    print(f"[INFO] load src+tgt test from: {src_tgt_train_file_name}.")

    timestamp = get_timestamp()
    fg_tag = "census_no_ad_w_fg" if apply_feature_group else "census_no_ad_wo_fg"
    date = get_current_date() + "_" + data_tag + "_" + fg_tag
    tries = 1
    task_id_list = list()
    for version in range(tries):
        hyperparameter_dict = {"lr": lr, "bs": batch_size, "ts": timestamp, "ve": version}
        task_id = date + "_" + train_data_tag + "_" + create_id_from_hyperparameters(hyperparameter_dict)
        task_id_list.append(task_id)
        print("[INFO] perform task:{0}".format(task_id))

        print("[INFO] model created.")
        src_train_loader, _ = get_income_census_dataloaders(
            ds_file_name=data_file_name_dict[train_data_tag], batch_size=batch_size, split_ratio=1.0)
        src_valid_loader, _ = get_income_census_dataloaders(
            ds_file_name=source_valid_file_name, batch_size=batch_size * 4, split_ratio=1.0)

        tgt_train_loader, _ = get_income_census_dataloaders(
            ds_file_name=target_ft_train_file_name, batch_size=batch_size, split_ratio=1.0)
        tgt_valid_loader, _ = get_income_census_dataloaders(
            ds_file_name=target_ft_valid_file_name, batch_size=batch_size * 4, split_ratio=1.0)
        tgt_test_loader, _ = get_income_census_dataloaders(
            ds_file_name=target_ft_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)

        plat = FederatedDAANLearner(model=model,
                                    source_da_train_loader=src_train_loader,
                                    source_val_loader=src_valid_loader,
                                    target_da_train_loader=tgt_train_loader,
                                    target_val_loader=tgt_valid_loader,
                                    epoch_patience=epoch_patience,
                                    validation_batch_interval=5)
        plat.set_model_save_info(census_no_ad_root_dir)

        plat.train_wo_adaption(epochs=max_epochs,
                               lr=lr,
                               train_source=True,
                               metric=valid_metric,
                               task_id=task_id,
                               momentum=momentum,
                               weight_decay=weight_decay)

        # load best model
        model.load_model(root=census_no_ad_root_dir,
                         task_id=task_id,
                         load_global_classifier=True,
                         timestamp=None)

        print("[DEBUG] Global classifier Model Parameter After train:")
        model.print_parameters()

        acc, auc, ks = test_classifier(model, tgt_test_loader, 'test')
        print(f"acc:{acc}, auc:{auc}, ks:{ks}")

    return task_id_list
