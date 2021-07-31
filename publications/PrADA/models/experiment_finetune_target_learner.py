import torch
import torch.optim as optim

from utils import get_timestamp, test_classifier, save_dann_experiment_result


class FederatedTargetLearner(object):

    def __init__(self, model, target_train_loader, target_val_loader, patience=200, max_global_epochs=500):
        self.model = model
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.patience = patience
        self.patient_count = None
        self.stop_training = None
        self.best_score = None
        self.timestamp_with_best_score = None
        self.root = "census_target"
        self.task_meta_file_name = "task_meta"
        self.task_id = None
        self.max_global_epochs = max_global_epochs

    def set_model_save_info(self, model_root):
        self.root = model_root

    def _check_exists(self):
        if self.model.check_discriminator_exists() is False:
            raise RuntimeError('Discriminator not set.')

    def _change_to_train_mode(self):
        self.model.change_to_train_mode()

    def _change_to_eval_mode(self):
        self.model.change_to_eval_mode()

    def save_model(self, task_id=None, timestamp=None):
        """Save trained model."""
        self.model.save_model(self.root, task_id, self.task_meta_file_name, timestamp=timestamp)

    def train_target_models(self,
                            epochs,
                            optimizer,
                            lr,
                            curr_global_epoch=0,
                            global_epochs=1,
                            metric=('ks', 'auc')):
        # num_batch = len(self.target_train_loader)
        loss = list()
        curr_lr = lr
        for ep in range(epochs):
            # start_steps = curr_global_epoch * epochs * num_batch + ep * num_batch
            # total_steps = self.max_global_epochs * epochs * num_batch

            for batch_idx, (batch_data, batch_label) in enumerate(self.target_train_loader):
                self._change_to_train_mode()

                # p = float(batch_idx + start_steps) / total_steps
                # curr_lr = adjust_learning_rate(optimizer, p, lr_0=lr, beta=0.75)
                # print(f"curr_global_epoch:{curr_global_epoch}, epochs:{epochs}")
                # print(f"total_steps:{total_steps};start_steps:{start_steps};batch_idx:{batch_idx};p:{p};lr:{curr_lr}")

                class_loss = self.model.compute_classification_loss(batch_data, batch_label)
                class_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(class_loss.item())
                with torch.no_grad():
                    if (batch_idx + 1) % 5 == 0:
                        self._change_to_eval_mode()
                        tgt_cls_acc, tgt_cls_auc, tgt_cls_ks = test_classifier(self.model,
                                                                               self.target_val_loader,
                                                                               "test target")
                        batch_per_epoch = 100. * batch_idx / len(self.target_train_loader)
                        print(f'[INFO] [{curr_global_epoch}/{global_epochs}]\t [{ep}/{epochs} ({batch_per_epoch:.0f}%)]'
                              f'\t loss:{class_loss}\t val target acc: {tgt_cls_acc:.6f}')
                        print(f'current learning rate:{curr_lr}')

                        metric_dict = {'acc': tgt_cls_acc, 'auc': tgt_cls_auc, 'ks': tgt_cls_ks}
                        score_list = [metric_dict[metric_name] for metric_name in metric]
                        score = sum(score_list) / len(score_list)
                        if score > self.best_score:
                            self.best_score = score
                            print(f"best score:{self.best_score} "
                                  f"with TARGET: acc:{tgt_cls_acc}, auc:{tgt_cls_auc}, ks:{tgt_cls_ks}")
                            param_dict = self.model.get_global_classifier_parameters()
                            metric_dict = dict()
                            metric_dict["target_cls_acc"] = tgt_cls_acc
                            metric_dict["target_cls_auc"] = tgt_cls_auc
                            metric_dict["target_cls_ks"] = tgt_cls_ks
                            timestamp = get_timestamp()
                            self.timestamp_with_best_score = timestamp
                            save_dann_experiment_result(self.root, self.task_id, param_dict, metric_dict, timestamp)
                            self.save_model(self.task_id, timestamp=timestamp)
                            print("saved last model")
                            self.patient_count = 0
                        else:
                            self.patient_count += 1
                            if self.patient_count > self.patience:
                                print(
                                    f"[INFO] Early Stopped at target_cls_acc:{self.best_score} "
                                    f"with timestamp:{self.timestamp_with_best_score}")
                                self.stop_training = True
                                break

            if self.stop_training:
                break

        return loss

    def train_target_with_alternating(self,
                                      global_epochs,
                                      top_epochs,
                                      bottom_epochs,
                                      lr,
                                      task_id,
                                      metric=('ks', 'auc'),
                                      momentum=0.99,
                                      weight_decay=0.00001):
        self.task_id = task_id
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        curr_lr = lr
        # step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        self.model.print_parameters()

        loss_list = list()
        self.patient_count = 0
        self.stop_training = False
        self.best_score = -float('inf')
        for ep in range(global_epochs):

            print(f"[INFO] ===> global epoch {ep}, start fine-tuning top")
            self.model.freeze_source_classifier(is_freeze=False)
            self.model.freeze_bottom(is_freeze=True)
            loss_list += self.train_target_models(top_epochs,
                                                  optimizer,
                                                  curr_lr,
                                                  curr_global_epoch=ep,
                                                  global_epochs=global_epochs,
                                                  metric=metric)

            print(f"[INFO] ===> global epoch {ep}, start fine-tuning bottom")
            self.model.freeze_source_classifier(is_freeze=False)
            self.model.freeze_bottom(is_freeze=True)
            loss_list += self.train_target_models(bottom_epochs,
                                                  optimizer,
                                                  curr_lr,
                                                  curr_global_epoch=ep,
                                                  global_epochs=global_epochs,
                                                  metric=metric)
            # step_lr.step()
            # curr_lr = step_lr.get_last_lr()
            print("[INFO] change learning rate to {0}".format(curr_lr))

            if self.stop_training:
                break

    # def train_target_as_whole(self, global_epochs, lr, task_id, dann_exp_result=None):
    #     self.task_id = task_id
    #
    #     print("[INFO] Do not apply DANN lr parameters")
    #     optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    #
    #     self.patient_count = 0
    #     self.stop_training = False
    #     self.best_score = -float('inf')
    #     self.model.freeze_bottom(is_freeze=True)
    #     # self.wrapper.freeze_bottom_aggregators(is_freeze=True)
    #     # self.wrapper.freeze_bottom(is_freeze=False, region_idx_list=self.fine_tuning_region_idx_list)
    #     # self.wrapper.print_parameters(print_all=True)
    #
    #     # self.wrapper.freeze_bottom(is_freeze=True, region_idx_list=[0, 1, 3, 5, 6, 8])
    #     # self.wrapper.freeze_bottom(is_freeze=True, region_idx_list=[0, 1, 3, 4, 5, 6, 7, 8])
    #     # self.wrapper.freeze_bottom_extractors(is_freeze=True)
    #     for ep in range(global_epochs):
    #         self.train_target_models(epochs=1, optimizer=optimizer, lr=lr,
    #                                  curr_global_epoch=ep, global_epochs=global_epochs)
