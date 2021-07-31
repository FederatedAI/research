from data_process.ppd_process.ppd_data_creation_config import ppd_data_creation

feature_extractor_architecture_list = [
    [15, 20, 15, 6],
    [85, 100, 60, 8],
    [30, 50, 30, 6],
    [18, 30, 18, 6],
    [55, 70, 30, 8]]

no_fg_feature_extractor_architecture = [203, 210, 70, 20]

pre_train_hyperparameters = {
    "using_interaction": False,
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 64,
    "max_epochs": 600,
    "epoch_patience": 3,
    "pos_class_weight": 3.0,
    "valid_metric": ('ks', 'auc')
}

fine_tune_hyperparameters = {
    "using_interaction": False,
    "load_global_classifier": False,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr": 6e-4,
    "batch_size": 64,
    "pos_class_weight": 1.0,
    "valid_metric": ('ks', 'auc')
}

no_adaptation_hyperparameters = {
    "apply_feature_group": False,
    "train_data_tag": 'all',  # can be either 'all' or 'tgt'
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 64,
    "max_epochs": 600,
    "epoch_patience": 3,
    "valid_metric": ('ks', 'auc'),
    "pos_class_weight": 3.0
}

data_dir = ppd_data_creation['processed_data_dir']
# data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_20210522/"
# data_tag = 'YOUR DATA TAG'
data_tag = 'all4000pos004'

data_hyperparameters = {
    "source_ad_train_file_name": data_dir + f"PPD_2014_src_1to9_ad_{data_tag}_train.csv",
    "source_ad_valid_file_name": data_dir + f'PPD_2014_src_1to9_ad_{data_tag}_valid.csv',
    "src_tgt_train_file_name": data_dir + f"PPD_2014_src_tgt_{data_tag}_train.csv",

    "target_ad_train_file_name": data_dir + f'PPD_2014_tgt_10to12_ad_{data_tag}_train.csv',
    "target_ft_train_file_name": data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_train.csv',
    "target_ft_valid_file_name": data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_valid.csv',
    "target_ft_test_file_name": data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv',

    "ppd_fg_pretrained_model_dir": "ppd_fg_pretrained_model",
    "ppd_fg_ft_target_model_dir": "ppd_fg_ft_target_model",
    "ppd_no-fg_pretrained_model_dir": "ppd_no-fg_pretrained_model",
    "ppd_no-fg_ft_target_model_dir": "ppd_no-fg_ft_target_model",
    "ppd_no-ad_model_dir": "ppd_no-ad_model"
}
