import argparse

from experiments.income_census.train_config import fine_tune_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_fg_adapt_pretrain import create_fg_census_global_model
from experiments.income_census.train_census_utils import finetune_census


def get_finetune_model_meta():
    finetune_target_root_dir = data_hyperparameters['census_fg_ft_target_model_dir']
    using_interaction = fine_tune_hyperparameters['using_interaction']
    model = create_fg_census_global_model(using_interaction=using_interaction)
    return model, finetune_target_root_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser("census_fg_target_fine_tune")
    parser.add_argument('--pretrain_task_id', type=str)
    args = parser.parse_args()
    pretrain_task_id = args.pretrain_task_id
    print(f"[INFO] fine-tune pre-trained model with pretrain task id : {pretrain_task_id}")

    census_pretain_model_root_dir = data_hyperparameters['census_fg_pretrained_model_dir']
    init_model, census_finetune_target_model_root_dir = get_finetune_model_meta()
    task_id = finetune_census(pretrain_task_id,
                              census_pretain_model_root_dir,
                              census_finetune_target_model_root_dir,
                              fine_tune_hyperparameters,
                              data_hyperparameters,
                              init_model)
    print(f"[INFO] finetune task id:{task_id}")
