from experiments.income_census.train_config import data_tag, no_adaptation_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_fg_adapt_pretrain import create_fg_census_global_model
from experiments.income_census.train_census_no_fg_adapt_pretrain import create_no_fg_census_global_model
from experiments.income_census.train_census_utils import train_no_adaptation


def get_model_meta():
    no_da_root_dir = data_hyperparameters["census_no-ad_model_dir"]
    apply_feature_group = no_adaptation_hyperparameters['apply_feature_group']
    if apply_feature_group:
        print("[INFO] feature grouping applied")
        model = create_fg_census_global_model(num_wide_feature=5)
    else:
        print("[INFO] no feature grouping applied")
        model = create_no_fg_census_global_model(aggregation_dim=4, num_wide_feature=5)
    return model, no_da_root_dir


if __name__ == "__main__":
    init_model, census_no_ad_root_dir = get_model_meta()
    task_id_list = train_no_adaptation(data_tag,
                                       census_no_ad_root_dir,
                                       no_adaptation_hyperparameters,
                                       data_hyperparameters,
                                       init_model)
    print(f"[INFO] task id list:{task_id_list}")
