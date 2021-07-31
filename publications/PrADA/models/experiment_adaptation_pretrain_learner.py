import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from utils import get_timestamp, save_dann_experiment_result
from utils import test_classifier, test_discriminator


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def adjust_learning_rate(optimizer, p, lr_0, beta=0.75):
    alpha = 10
    lr = lr_0 / (1 + alpha * p) ** beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class FederatedDAANLearner(object):

    def __init__(self,
                 model,
                 source_da_train_loader,
                 source_val_loader,
                 target_da_train_loader=None,
                 target_val_loader=None,
                 max_epochs=500,
                 epoch_patience=2,
                 validation_batch_interval=10,
                 number_validations=None):
        self.global_model = model
        self.num_regions = self.global_model.get_num_regions()
        self.src_train_loader = source_da_train_loader
        self.tgt_train_loader = target_da_train_loader
        # self.tgt_clz_train_loader = target_clz_train_loader

        self.src_val_loader = source_val_loader
        self.tgt_val_loader = target_val_loader

        if self.src_train_loader: print(f"source_train_loader len: {len(self.src_train_loader)}")
        if self.src_val_loader: print(f"source_val_loader len: {len(self.src_val_loader)}")
        if self.tgt_train_loader: print(f"target_train_loader len: {len(self.tgt_train_loader)}")
        if self.tgt_val_loader: print(f"target_val_loader len: {len(self.tgt_val_loader)}")

        self.max_epochs = max_epochs
        self.epoch_patience = epoch_patience
        self.root = "census_dann"
        self.task_meta_file_name = "task_meta"
        self.best_score = None
        self.recoded_timestamp = None
        self.num_validations = number_validations
        self.valid_batch_interval = validation_batch_interval
        self.stop_training = False

    def set_model_save_info(self, root):
        self.root = root

    def _check_exists(self):
        if self.global_model.check_discriminator_exists() is False:
            raise RuntimeError('Discriminator not set.')

    def _change_to_train_mode(self):
        self.global_model.change_to_train_mode()

    def _change_to_eval_mode(self):
        self.global_model.change_to_eval_mode()

    def save_model(self, task_id, timestamp):
        self.global_model.save_model(self.root, task_id, self.task_meta_file_name, timestamp=timestamp)

    def train_wo_adaption(self,
                          epochs,
                          lr,
                          task_id,
                          train_source=True,
                          valid_source=False,
                          metric=('ks', 'auc'),
                          momentum=0.99,
                          weight_decay=0.00001):

        optimizer = optim.SGD(self.global_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        train_loader = self.src_train_loader if train_source else self.tgt_train_loader
        num_batches_per_epoch = len(train_loader)
        num_validations, valid_batch_interval = self.compute_number_validations(num_batches_per_epoch)
        val_loader_dict = {"src": self.src_val_loader, "tgt": self.tgt_val_loader} if train_source else {
            "tgt": self.tgt_val_loader}

        validation_patience_count = 0
        self.stop_training = False
        self.best_score = -float('inf')
        for ep in range(epochs):
            for batch_idx, (batch_data, batch_label) in enumerate(train_loader):
                self._change_to_train_mode()
                class_loss = self.global_model.compute_classification_loss(batch_data, batch_label)

                class_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    if (batch_idx + 1) % valid_batch_interval == 0:
                        self._change_to_eval_mode()

                        metric_dict = dict()
                        result_list = list()
                        for domain, val_loader in val_loader_dict.items():
                            acc, auc, ks = test_classifier(self.global_model, val_loader, 'valid')
                            result_list.append(f"{domain} - acc:{acc}, auc:{auc}, ks:{ks} \n")
                            metric_dict[domain] = {'acc': acc, 'auc': auc, 'ks': ks}

                        batch_per_epoch = 100. * batch_idx / num_batches_per_epoch
                        result = ";".join(result_list)
                        print(f'[INFO] [{ep}/{epochs} ({batch_per_epoch:.0f}%)]\t loss:{class_loss.item()}\t')
                        print(f"[INFO] {result}")

                        valid_metrics = metric_dict['src'] if valid_source else metric_dict['tgt']
                        score_list = [valid_metrics[metric_name] for metric_name in metric]
                        score = sum(score_list) / len(score_list)
                        print(f"[DEBUG] *score: {score}")
                        if score > self.best_score:
                            self.best_score = score
                            print(f"best auc:{self.best_score}.")
                            param_dict = self.global_model.get_global_classifier_parameters()
                            timestamp = get_timestamp()
                            self.recoded_timestamp = timestamp
                            save_dann_experiment_result(self.root, task_id, param_dict, metric_dict, timestamp)
                            self.save_model(task_id, timestamp)
                            validation_patience_count = 0
                        else:
                            validation_patience_count += 1
                            epoch_patience_count = self.compute_epoch_patience_count(validation_patience_count,
                                                                                     num_validations)
                            if epoch_patience_count >= self.epoch_patience:
                                print(
                                    f"[INFO] Early Stopped at epoch:{ep}, batch:{batch_idx} with target_cls_acc:"
                                    f"{self.best_score}")
                                print(
                                    f"[INFO] Models with best score stored at timestamp:{self.recoded_timestamp}")
                                self.stop_training = True
                                break

            if self.stop_training:
                break

    def compute_number_validations(self, num_batches_per_epoch):
        assert self.valid_batch_interval is not None or self.num_validations is not None

        valid_batch_interval = self.valid_batch_interval
        num_validations = self.num_validations
        if num_validations is None:
            num_validations = int(num_batches_per_epoch / valid_batch_interval)
        else:
            valid_batch_interval = int(num_batches_per_epoch / num_validations)

        print("[INFO] num_batches_per_epoch:{0}".format(num_batches_per_epoch))
        print("[INFO] valid_batch_interval:{0}".format(valid_batch_interval))
        print("[INFO] number_validations:{0}".format(num_validations))
        return num_validations, valid_batch_interval

    def compute_epoch_patience_count(self, validation_patience_count, num_validations_per_epoch):
        epoch_patience_count = float(validation_patience_count) / num_validations_per_epoch
        epoch_patience_ratio = 100 * (epoch_patience_count / self.epoch_patience)
        print(f"[DEBUG] current epoch patience count:{epoch_patience_count}")
        print(f"[DEBUG] current epoch patience ratio:{epoch_patience_ratio:.1f}%")
        return epoch_patience_count

    @staticmethod
    def get_optimizer_params(optimizer_param_dict, optimizer_domain_tag):
        lr = optimizer_param_dict[optimizer_domain_tag]["lr"]
        momentum = optimizer_param_dict[optimizer_domain_tag]["momentum"]
        weight_decay = optimizer_param_dict[optimizer_domain_tag]["weight_decay"]
        return lr, momentum, weight_decay

    def train_dann(self,
                   epochs,
                   task_id,
                   metric=('ks', 'auc'),
                   optimizer_param_dict=None,
                   monitor_source=False):
        self._check_exists()

        src_da_train_iter = ForeverDataIterator(self.src_train_loader)
        tgt_da_train_iter = ForeverDataIterator(self.tgt_train_loader)

        src_lr, src_momentum, src_weight_decay = self.get_optimizer_params(optimizer_param_dict, "src")
        src_optimizer = optim.SGD(self.global_model.parameters(),
                                  lr=src_lr,
                                  momentum=src_momentum,
                                  weight_decay=src_weight_decay)

        num_batches_per_epoch = len(src_da_train_iter)
        num_validations, valid_batch_interval = self.compute_number_validations(num_batches_per_epoch)

        validation_patience_count = 0
        self.stop_training = False
        self.best_score = -float('inf')
        target_auc_list = []
        for ep in range(epochs):
            start_steps = ep * num_batches_per_epoch
            total_steps = self.max_epochs * num_batches_per_epoch
            for batch_idx in range(num_batches_per_epoch):
                self._change_to_train_mode()
                source_data, source_label = next(src_da_train_iter)
                target_data, target_label = next(tgt_da_train_iter)

                p = float(batch_idx + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                curr_lr = adjust_learning_rate(src_optimizer, p, lr_0=src_lr)

                print(f"[DEBUG] alpha:{alpha}")

                # source has domain label of zero, while target has domain label of one
                domain_source_labels = torch.zeros(source_label.shape[0]).long()
                domain_target_labels = torch.ones(target_label.shape[0]).long()

                kwargs = dict()
                kwargs["alpha"] = alpha
                loss_dict = self.global_model.compute_total_loss(source_data,
                                                                 target_data,
                                                                 source_label,
                                                                 target_label,
                                                                 domain_source_labels,
                                                                 domain_target_labels,
                                                                 **kwargs)

                # back-propagation and optimization
                total_loss = loss_dict["src_total_loss"]
                total_loss.backward()
                src_optimizer.step()
                src_optimizer.zero_grad()

                with torch.no_grad():
                    if (batch_idx + 1) % valid_batch_interval == 0:
                        print("-" * 50)
                        self._change_to_eval_mode()
                        src_cls_acc, src_cls_auc, src_cls_ks = test_classifier(model=self.global_model,
                                                                               data_loader=self.src_val_loader,
                                                                               tag="test source")
                        tgt_cls_acc, tgt_cls_auc, tgt_cls_ks = test_classifier(model=self.global_model,
                                                                               data_loader=self.tgt_val_loader,
                                                                               tag="test target")
                        ave_dom_acc, dom_acc_list, entropy_dom_acc = test_discriminator(self.global_model,
                                                                                        self.num_regions,
                                                                                        self.src_val_loader,
                                                                                        self.tgt_val_loader)
                        total_dom_acc, source_dom_acc, target_dom_acc = ave_dom_acc
                        total_acc_list, source_acc_list, target_acc_list = dom_acc_list
                        batch_per_epoch = 100. * batch_idx / len(self.src_train_loader)
                        print(f"[DEBUG] current learning rate:{curr_lr}")
                        print(f'[INFO] batch_idx:{batch_idx}')
                        print(f'[INFO] [{ep}/{epochs} ({batch_per_epoch:.0f}%)]\t loss:{total_loss}')
                        print(f'[INFO] val src cls acc: {src_cls_acc:.6f} \t val tgt cls acc: {tgt_cls_acc:.6f}')
                        print(f'[INFO] val src cls auc: {src_cls_auc:.6f} \t val tgt cls auc: {tgt_cls_auc:.6f}')
                        print('[INFO] val src dom acc: {:.6f} \t val tgt dom acc: {:.6f}\t '
                              'val total dom acc: {:.6f}'.format(source_dom_acc, target_dom_acc, total_dom_acc))
                        print(f"[DEBUG] current lr: {curr_lr}")
                        print(f"[DEBUG] current alpha: {alpha}")
                        target_auc_list.append(tgt_cls_auc)

                        if monitor_source:
                            metric_dict = {'acc': src_cls_acc, 'auc': src_cls_auc, 'ks': src_cls_ks}
                        else:
                            metric_dict = {'acc': tgt_cls_acc, 'auc': tgt_cls_auc, 'ks': tgt_cls_ks}
                        score_list = [metric_dict[metric_name] for metric_name in metric]
                        metric_score = sum(score_list) / len(score_list)

                        score = metric_score
                        print(f"[DEBUG] *total_score: {score}")
                        if score > self.best_score:
                            self.best_score = score
                            print(f"[INFO] best score:{self.best_score}")
                            print(f"[INFO] SOURCE: acc:{src_cls_acc}, auc:{src_cls_auc}, ks:{src_cls_ks}")
                            print(f"[INFO] TARGET: acc:{tgt_cls_acc}, auc:{tgt_cls_auc}, ks:{tgt_cls_ks}")
                            param_dict = self.global_model.get_global_classifier_parameters()
                            metric_dict = dict()
                            metric_dict["source_cls_acc"] = src_cls_acc
                            metric_dict["source_cls_auc"] = src_cls_auc
                            metric_dict["source_cls_ks"] = src_cls_ks
                            metric_dict["target_cls_acc"] = tgt_cls_acc
                            metric_dict["target_cls_auc"] = tgt_cls_auc
                            metric_dict["target_cls_ks"] = tgt_cls_ks
                            metric_dict["source_dom_acc"] = source_dom_acc
                            metric_dict["target_dom_acc"] = target_dom_acc
                            metric_dict["source_dom_acc"] = source_dom_acc
                            metric_dict["target_dom_acc"] = target_dom_acc
                            metric_dict["source_dom_acc_list"] = source_acc_list
                            metric_dict["target_dom_acc_list"] = target_acc_list
                            metric_dict["target_batch_auc_list"] = target_auc_list
                            metric_dict["current_epoch"] = ep
                            metric_dict["current_batch_idx"] = batch_idx
                            metric_dict["num_batches_per_epoch"] = num_batches_per_epoch
                            metric_dict["num_validations"] = num_validations
                            metric_dict["valid_batch_interval"] = valid_batch_interval
                            timestamp = get_timestamp()
                            self.recoded_timestamp = timestamp
                            save_dann_experiment_result(self.root, task_id, param_dict, metric_dict, timestamp)
                            self.save_model(task_id, timestamp)
                            # reset patience count
                            validation_patience_count = 0
                        else:
                            validation_patience_count += 1
                            epoch_patience_count = self.compute_epoch_patience_count(validation_patience_count,
                                                                                     num_validations)
                            if epoch_patience_count >= self.epoch_patience:
                                print(
                                    f"[INFO] Early Stopped at epoch:{ep}, batch:{batch_idx} with target_cls_acc:"
                                    f"{self.best_score}")
                                print(
                                    f"[INFO] Models with best score stored at timestamp:{self.recoded_timestamp}")
                                self.stop_training = True
                                break

            if self.stop_training:
                break
