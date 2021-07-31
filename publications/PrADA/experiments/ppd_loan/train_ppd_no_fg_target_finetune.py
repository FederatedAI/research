import argparse

from experiments.ppd_loan.train_config import data_hyperparameters, fine_tune_hyperparameters
from experiments.ppd_loan.train_ppd_no_fg_adapt_pretrain import create_no_fg_pdd_global_model
from experiments.ppd_loan.train_ppd_utils import finetune_ppd


def get_finetune_model_meta():
    finetune_target_root_dir = data_hyperparameters['ppd_no-fg_ft_target_model_dir']
    pos_class_weight = fine_tune_hyperparameters['pos_class_weight']
    model = create_no_fg_pdd_global_model(pos_class_weight=pos_class_weight)
    return model, finetune_target_root_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ppd_no-fg_target_fine_tune")
    parser.add_argument('--pretrain_task_id', type=str)
    args = parser.parse_args()
    pretrain_task_id = args.pretrain_task_id
    print(f"[INFO] fine-tune pre-trained model with pretrain task id : {pretrain_task_id}")

    ppd_pretain_model_root_dir = data_hyperparameters['ppd_no-fg_pretrained_model_dir']
    init_model, ppd_finetune_target_model_root_dir = get_finetune_model_meta()

    task_id = finetune_ppd(pretrain_task_id,
                           ppd_pretain_model_root_dir,
                           ppd_finetune_target_model_root_dir,
                           fine_tune_hyperparameters,
                           data_hyperparameters,
                           model=init_model)
    print(f"[INFO] task id:{task_id}")
