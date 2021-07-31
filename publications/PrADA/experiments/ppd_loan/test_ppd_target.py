import argparse

from datasets.ppd_dataloader import get_pdd_dataloaders
from experiments.ppd_loan import train_ppd_fg_target_finetune as fg_finetune
from experiments.ppd_loan import train_ppd_no_adaptation as no_ad_finetune
from experiments.ppd_loan import train_ppd_no_fg_target_finetune as no_fg_finetune
from experiments.ppd_loan.train_config import data_hyperparameters
from experiments.test_utils import test_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser("ppd_test_target")
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--model_tag', type=str)
    args = parser.parse_args()
    task_id = args.task_id
    model_tag = args.model_tag
    print(f"[INFO] perform test task on : [{model_tag}] with id: {task_id}")
    test_models_dir = {"fg": fg_finetune.get_finetune_model_meta,
                       "no_fg": no_fg_finetune.get_finetune_model_meta,
                       "no_ad": no_ad_finetune.get_model_meta}
    init_model, model_root_dir = test_models_dir[model_tag]()
    target_test_file_name = data_hyperparameters['target_ft_test_file_name']
    print(f"[INFO] target_test_file_name: {target_test_file_name}.")

    print("[INFO] load test data")
    target_test_loader, _ = get_pdd_dataloaders(ds_file_name=target_test_file_name,
                                                batch_size=1024,
                                                split_ratio=1.0)

    test_model(task_id, init_model, model_root_dir, target_test_loader)
