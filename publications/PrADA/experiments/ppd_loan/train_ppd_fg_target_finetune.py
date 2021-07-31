from experiments.ppd_loan.train_config import data_hyperparameters, fine_tune_hyperparameters
from experiments.ppd_loan.train_ppd_fg_adapt_pretrain import create_fg_pdd_global_model
from experiments.ppd_loan.train_ppd_utils import finetune_ppd
import argparse


def get_finetune_model_meta():
    finetune_target_root_dir = data_hyperparameters['ppd_fg_ft_target_model_dir']
    using_interaction = fine_tune_hyperparameters['using_interaction']
    pos_class_weight = fine_tune_hyperparameters['pos_class_weight']
    model = create_fg_pdd_global_model(pos_class_weight=pos_class_weight,
                                       using_interaction=using_interaction)
    return model, finetune_target_root_dir


if __name__ == "__main__":

    # parser = argparse.ArgumentParser("ppd_fg_target_fine_tune")
    # parser.add_argument('--pretrain_task_id', type=str)
    # args = parser.parse_args()
    # pretrain_task_id = args.pretrain_task_id
    pretrain_task_id = "20210731_ppd_fg_adapt_all4000pos004_intrFalse_pw1.0_lr0.0005_bs64_me600_ts1627666700"
    print(f"[INFO] fine-tune pre-trained model with pretrain task id : {pretrain_task_id}")

    ppd_pretain_model_root_dir = data_hyperparameters['ppd_fg_pretrained_model_dir']
    init_model, ppd_finetune_target_model_root_dir = get_finetune_model_meta()

    task_id = finetune_ppd(pretrain_task_id,
                           ppd_pretain_model_root_dir,
                           ppd_finetune_target_model_root_dir,
                           fine_tune_hyperparameters,
                           data_hyperparameters,
                           model=init_model)
    print(f"[INFO] task id:{task_id}")
