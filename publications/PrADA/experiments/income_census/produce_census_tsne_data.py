import argparse

from experiments.income_census.tsne_config import tsne_embedding_creation
from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census import train_census_fg_adapt_pretrain as fg_dann
from experiments.income_census import train_census_no_fg_adapt_pretrain as no_fg_dann
from experiments.income_census.train_config import data_tag, data_dir, data_hyperparameters
from utils import produce_data_for_distribution

if __name__ == "__main__":

    parser = argparse.ArgumentParser("census_tsne_distribution")
    parser.add_argument('--task_id', type=str, required=True)
    # parser.add_argument('--adapt', default=False, action='store_true')
    # parser.add_argument('--interaction', default=False, action='store_true')
    args = parser.parse_args()
    task_id = args.task_id
    apply_adaptation = tsne_embedding_creation["apply_adaptation"]
    using_interaction = tsne_embedding_creation["using_interaction"]

    print(f"[INFO] task id : {task_id}")
    print(f"[INFO] apply adaptation : {apply_adaptation}")
    print(f"[INFO] using interaction : {using_interaction}")

    if apply_adaptation:
        tag = "ad"
        model_dir = data_hyperparameters["census_fg_pretrained_model_dir"]
    else:
        tag = "no-ad"
        model_dir = data_hyperparameters["census_no-ad_model_dir"]

    feature_group_name_list = ['employment', 'demographics', 'migration', 'household']
    if using_interaction:
        feature_grp_intr_name_list = ['emp-demo', 'emp-mig', 'emp-house', 'demo-mig', 'demo-house', 'mig-house']
        feature_group_name_list = feature_group_name_list + feature_grp_intr_name_list

    tsne_embedding_dir = tsne_embedding_creation["tsne_embedding_data_dir"]

    source_train_file_name = data_dir + f'undergrad_census9495_ad_{data_tag}_train.csv'
    target_train_file_name = data_dir + f'grad_census9495_ad_{data_tag}_train.csv'

    # load pre-trained model
    print("[INFO] load pre-trained model.")
    use_feature_group = True
    if use_feature_group:
        model = fg_dann.create_fg_census_global_model(using_interaction=using_interaction)
    else:
        model = no_fg_dann.create_no_fg_census_global_model()

    model.load_model(root=model_dir,
                     task_id=task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[INFO] load data.")
    batch_size = 4000
    target_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)
    source_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=source_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] produce data for drawing TSNE feature distribution.")
    produce_data_for_distribution(model,
                                  source_train_loader,
                                  target_train_loader,
                                  feature_group_name_list,
                                  tsne_embedding_dir,
                                  tag + str(batch_size))
