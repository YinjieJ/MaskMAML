import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import IFCNReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule

from utils.experiment import pearson_score

import copy


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLRegressor(nn.Module):
    def __init__(self, input_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLRegressor, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.current_epoch = 0
        self.input_shape = input_shape

        self.rng = set_torch_seed(seed=args.seed)

        self.args.rng = self.rng
        self.regressor = IFCNReLUNormNetwork(input_shape=self.input_shape, args=self.args,
                                            device=device, meta=True).to(device=self.device)

        self.task_learning_rate = args.task_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate)
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(params=self.regressor.named_parameters()))

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)
        self.optimizer_adapt = optim.Adam(self.trainable_adaptation_parameters(), lr=args.meta_learning_rate,
                                          amsgrad=False)
        self.scheduler_adapt = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_adapt,
                                                                    T_max=self.args.total_epochs,
                                                                    eta_min=self.args.min_learning_rate)

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    # if "norm_layer" not in name:
                    #    param_dict[name] = param.to(device=self.device)

                    if "layer_dict.linear.bias" in name or "layer_dict.linear.weights" in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        self.regressor.zero_grad(names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order)
        names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_wrt_params,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses[0]))
        losses['loss_supp'] = torch.mean(torch.stack(total_losses[1]))
        losses['loss_0'] = torch.mean(torch.stack(total_losses[2]))
        losses['loss_supp_0'] = torch.mean(torch.stack(total_losses[3]))
        if sum(~np.isnan(total_accuracies[0])) == 0:
            losses['accuracy'] = 0
        else:
            losses['accuracy'] = np.mean(np.array(total_accuracies[0])[~np.isnan(total_accuracies[0])])
            losses['accuracy_supp'] = np.mean(np.array(total_accuracies[1])[~np.isnan(total_accuracies[1])])
            losses['accuracy_0'] = np.mean(np.array(total_accuracies[2])[~np.isnan(total_accuracies[2])])
            losses['accuracy_supp_0'] = np.mean(np.array(total_accuracies[3])[~np.isnan(total_accuracies[3])])

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        support_set_x, support_set_y, support_set_z, support_set_assay, \
        target_set_x, target_set_y, target_set_z, target_set_assay = data_batch

        b = len(support_set_y)

        total_losses = []
        total_accuracies = []
        total_losses_supp = []
        total_accuracies_supp = []
        total_losses_0 = []
        total_accuracies_0 = []
        total_losses_supp_0 = []
        total_accuracies_supp_0 = []
        total_losses_only = []
        per_task_target_preds = [[] for i in range(len(target_set_x))]
        self.regressor.zero_grad()

        for task_id, (support_set_x_task, support_set_y_task, support_set_z_task, support_set_assay_task,
                      target_set_x_task, target_set_y_task, target_set_z_task, target_set_assay_task,) in \
                enumerate(zip(support_set_x,
                              support_set_y,
                              support_set_z,
                              support_set_assay,
                              target_set_x,
                              target_set_y,
                              target_set_z,
                              target_set_assay)):

            # first of all, put all tensors to the device
            support_set_x_task = torch.Tensor(support_set_x_task[0]).float().to(device=self.device)
            support_set_y_task = torch.Tensor(support_set_y_task[0]).float().to(device=self.device)
            support_set_z_task = torch.IntTensor(support_set_z_task[0]).int().to(device=self.device)
            support_set_assay_task = torch.LongTensor(support_set_assay_task).int().to(device=self.device)
            target_set_x_task = torch.Tensor(target_set_x_task[0]).float().to(device=self.device)
            target_set_y_task = torch.Tensor(target_set_y_task[0]).float().to(device=self.device)
            target_set_z_task = torch.IntTensor(target_set_z_task[0]).int().to(device=self.device)
            target_set_assay_task = torch.LongTensor(target_set_assay_task).int().to(device=self.device)

            task_losses = []
            task_accuracies = []

            task_losses_supp = []
            task_accuracies_supp = []

            task_losses_0 = []
            task_accuracies_0 = []

            task_losses_supp_0 = []
            task_accuracies_supp_0 = []

            task_losses_only = []

            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.regressor.named_parameters())

            ns, fp_dim = support_set_x_task.shape
            nt, _ = target_set_x_task.shape

            r = np.random.uniform(-3,3)
            for num_step in range(num_steps):

                support_loss, support_preds = self.net_forward(x=support_set_x_task,
                                                               y=support_set_y_task+r,
                                                               weights=names_weights_copy,
                                                               backup_running_statistics=
                                                               True if (num_step == 0) else False,
                                                               training=True, num_step=num_step)  # , support=True)

                if num_step == 0:
                    target_loss, target_preds = self.net_forward(x=target_set_x_task,
                                                                 y=target_set_y_task+r,
                                                                 weights=names_weights_copy,
                                                                 backup_running_statistics=False,
                                                                 training=True, num_step=0)

                    task_losses_supp_0.append(support_loss)
                    task_losses_0.append(target_loss)

                    accuracy_0 = pearson_score(target_set_y_task.detach().cpu().numpy(),
                                               (target_preds-r).detach().cpu().numpy())

                    task_losses_0 = torch.sum(torch.stack(task_losses_0))
                    total_losses_0.append(task_losses_0)
                    total_accuracies_0.append(accuracy_0)

                    accuracy_supp_0 = pearson_score(support_set_y_task.detach().cpu().numpy(),
                                                    (support_preds-r).detach().cpu().numpy())
                    task_losses_supp_0 = torch.sum(torch.stack(task_losses_supp_0))
                    total_losses_supp_0.append(task_losses_supp_0)
                    total_accuracies_supp_0.append(accuracy_supp_0)

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

            support_loss, support_preds = self.net_forward(x=support_set_x_task,
                                                           y=support_set_y_task+r, weights=names_weights_copy,
                                                           backup_running_statistics=False, training=True,
                                                           num_step=num_steps - 1)

            target_loss, target_preds = self.net_forward(x=target_set_x_task,
                                                         y=target_set_y_task+r, weights=names_weights_copy,
                                                         backup_running_statistics=False, training=True,
                                                         num_step=num_steps - 1)
            task_losses.append(target_loss)
            task_losses_supp.append(support_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            # _, predicted = torch.max(target_preds.data, 1)

            accuracy = pearson_score(target_set_y_task.detach().cpu().numpy(), (target_preds-r).detach().cpu().numpy())
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.append(accuracy)


            supp_accuracy = pearson_score(support_set_y_task.detach().cpu().numpy(),
                                          (support_preds-r).detach().cpu().numpy())
            task_losses_supp = torch.sum(torch.stack(task_losses_supp))
            total_losses_supp.append(task_losses_supp)
            total_accuracies_supp.append(supp_accuracy)

            # pdb.set_trace()

            if not training_phase:
                self.regressor.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(
            total_losses=[total_losses, total_losses_supp, total_losses_0, total_losses_supp_0],
            total_accuracies=[total_accuracies, total_accuracies_supp, total_accuracies_0, total_accuracies_supp_0])

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, mixup=None, lam=None,
                    support=None):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """

        preds = self.regressor.forward(x=x, params=weights,
                                       training=training,
                                       backup_running_statistics=backup_running_statistics, num_step=num_step,
                                       mixup=mixup, lam=lam)

        if mixup is not None:
            npreds = preds.shape[0]
            preds = preds[:int(npreds / 2), :]

        # loss = F.cross_entropy(input=preds, target=y)
        # if support is not None:
        #    loss = F.mse_loss(input=preds * sample_weights, target=y.unsqueeze(dim=-1))
        # else:
        loss = F.mse_loss(input=preds, target=y.unsqueeze(dim=-1))

        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def trainable_feature_parameters(self):

        for name, param in self.named_parameters():
            if param.requires_grad and 'learning_rates' not in name:  # 'layer_dict.linear.bias' not in name and 'layer_dict.linear.weights' not in name and 'learning_rates' not in name:
                yield param

    def trainable_adaptation_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and (
                    'layer_dict.linear.bias' in name or 'layer_dict.linear.weights' in name or 'learning_rates' in name):
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.regressor.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        '''
        if epoch < 10:
            self.optimizer.zero_grad()
            losses['loss'].backward()
            self.optimizer.step()
        else:
            self.optimizer_adapt.zero_grad()
            losses['loss'].backward()
            self.optimizer_adapt.step()
        '''

        '''
        self.optimizer.zero_grad()

        losses['loss'].backward(retain_graph=True)
        grad_loss_backup = {}

        for name, param in self.named_parameters():
            if param.requires_grad and name not in grad_loss_backup and 'learning_rates' not in name and "layer_dict.linear.bias" not in name or "layer_dict.linear.weights" not in name:
                grad_loss_backup[name] = copy.deepcopy(param.grad)

        self.optimizer.zero_grad()


        losses['loss_only'].backward()
        for name, param in self.named_parameters():
            if param.requires_grad and 'learning_rates' not in name and "layer_dict.linear.bias" not in name or "layer_dict.linear.weights" not in name:
                param.grad = copy.deepcopy(grad_loss_backup[name])

        self.optimizer.step()
        '''
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.optimizer_adapt.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
