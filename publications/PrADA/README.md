# PrADA: Privacy-preserving Federated Adversarial Domain Adaption over Feature Group for Interpretability

We walk through steps of running experiments on Census Income data. Experiments on PPD loan default follow the same procedure.



## 0. Download Data

- Census Income Data

  - You can download census income data from https://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD).
  - More detailed feature description can be found at https://docs.1010data.com/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
  
- PPD Loan Default

  - You can download census income data from https://github.com/yankang18/ppd_data
  
  

## 1. Prepare Census Income Data

Update the `census_data_creation_config.py` file.   

```python
census_data_creation = {
    "original_data_dir": "YOUR ORIGINAL DATA DIR",
    "processed_data_dir": "YOUR PROCESSED DATA DIR",
    "train_data_file_name": "census-income.data",
    "test_data_file_name": "census-income.test",
    "positive_sample_ratio": 0.04,
    "number_target_samples": 4000,
    "data_tag": "all4000pos004"
}
```

Then, run:

```python
python census_prepare_data.py
```

This would produce data at the location you specified at `"processed_data_dir"`



## 2. Perform Experiments

In this section, we show the steps of running following four variants of **PrADA**

1. **PrADA**:  apply feature group (FG) based domain adversarial adaptation (DA) with feature group interaction (IR).
2. **PrADA w/o IR**: apply feature group based domain adversarial adaptation without feature group interaction
3. **PrADA w/o FG&IR**: apply domain adversarial adaptation, but without feature grouping and interaction
4. **PrADA w/o DA&FG&IR**: without domain adversarial adaptation, feature grouping, and feature group interaction.



### 2.1 Run PrADA

1. Go to directory: **prada/experiments/income_census/**
2. Configure hyperparameters using `train_config.py` file. 
   - The `using_interaction` must be set to `True`

```python
pre_train_hyperparameters = {
    "using_interaction": True,
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 6e-4,
    "batch_size": 128,
    "max_epochs": 600,
    "epoch_patience": 3,
    "valid_metric": ('ks', 'auc')
}

fine_tune_hyperparameters = {
    "using_interaction": True,
    "load_global_classifier": False,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr": 8e-4,
    "batch_size": 128,
    "valid_metric": ('ks', 'auc')
}
```



3. First run pretrain task:

```python
python train_census_fg_adapt_pretrain.py 
```

Once the training is completed, a pretrain task is returned, e.g.:

```
20210730_census_fg_adapt_all4000pos004_intrTrue_lr0.0005_bs128_me600_ts1627606557
```



4. Run finetune task with pretain task id as a input:

```
python train_census_fg_target_finetune.py --pretrain_task_id 20210730_census_fg_adapt_all4000pos004_intrTrue_lr0.0005_bs128_me600_ts1627606557
```

Output test AUC and test KS.

 

### 2.2 Run PrADA w/o IR

1. Go to directory: **prada/experiments/income_census/**
2. Configure hyperparameters using `train_config.py` file. 
   - The `using_interaction` must be set to `False`

```python
pre_train_hyperparameters = {
    "using_interaction": False,
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 128,
    "max_epochs": 600,
    "epoch_patience": 3,
    "valid_metric": ('ks', 'auc')
}

fine_tune_hyperparameters = {
    "using_interaction": False,
    "load_global_classifier": False,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr": 8e-4,
    "batch_size": 128,
    "valid_metric": ('ks', 'auc')
}
```



3. First run pretrain task:

```python
python train_census_fg_adapt_pretrain.py 
```

Once the training is completed, a task is returned, e.g.:

```
20210730_census_fg_adapt_all4000pos004_intrFalse_lr0.0005_bs128_me600_ts1627606557
```



4. Run finetune task with pretain task id as a input:

```
python train_census_fg_target_finetune.py --pretrain_task_id 20210730_census_fg_adapt_all4000pos004_intrFalse_lr0.0005_bs128_me600_ts1627606557
```

Output test AUC and test KS.



### 2.3 Run PrADA w/o FG&IR

1. Go to directory: **prada/experiments/income_census/**
2. Configure hyperparameters using `train_config.py` file. 
   - NOTE: the `using_interaction` will not be used in this setting because no feature group is applied. Therefore, leave it by default.

```python
pre_train_hyperparameters = {
    "using_interaction": False,
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 128,
    "max_epochs": 600,
    "epoch_patience": 3,
    "valid_metric": ('ks', 'auc')
}

fine_tune_hyperparameters = {
    "using_interaction": False,
    "load_global_classifier": False,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr": 8e-4,
    "batch_size": 128,
    "valid_metric": ('ks', 'auc')
}
```



3. First run pretrain task:

```Python
python train_census_no_fg_adapt_pretrain.py 
```

Once the training is completed, a task is returned, e.g.:

```
20210730_census_no_fg_adapt_all4000pos004_lr0.0005_bs128_me600_ts1627612696
```



4. Run finetune task with pretain task id as a input:

```
python train_census_no_fg_target_finetune.py --pretrain_task_id 20210730_census_no_fg_adapt_all4000pos004_lr0.0005_bs128_me600_ts1627612696
```

Output test AUC and test KS.



### 2.4 Run PrADA w/o DA&FG&IR

1. Go to directory: **prada/experiments/income_census/**
2. Configure hyperparameters using `train_config.py` file. 
   - the `train_data_tag` specifies whether you use all samples (source + target) or just target samples for training.
   - NOTE: `apply_feature_group` specifies whether applying feature grouping or not. In this setting, we always set it to `False`.

```Python
no_adaptation_hyperparameters = {
    "apply_feature_group": False,
    "train_data_tag": 'all',  # can be either 'all' or 'tgt'
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 128,
    "max_epochs": 600,
    "epoch_patience": 2,
    "valid_metric": ('ks', 'auc')
}
```



3. Run task:

```
python train_census_no_adaptation.py 
```

Once the training is completed, a task is returned, e.g.:

```
20210730_all4000pos004v8_census_no_ad_wo_fg_all_lr0.0005_bs128_ts1627613954_ve0/
```

Output test AUC and test KS.



### 2.5 Run Test

The test AUC and test KS on target test data will be given once the training is completed as shown above. You can also test the trained model on target test data with a separate command, shown as follows:

Go to directory: **prada/experiments/income_census/**

- `task_id` specifies the task of traing the model that you want to test.
- `model_tag` specifies the variant of PrADA model. It can be `fg` (with feature group), `no_fg` (with no feature group) or `no_ad` (with no adaptation and no feature group)

```
python test_census_target.py --task_id 20210731_census_fg_adapt_all4000pos004_intrFalse_lr0.0005_bs128_me600_ts1627682125@target_20210731_rt_glr_lr0.0008_bs128_ts1627683284 --model_tag fg
```

