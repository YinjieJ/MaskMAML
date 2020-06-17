import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from prune import Prune
from copy import deepcopy

class Meta(nn.Module):
    def __init__(self, args):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Prune()
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    # def clip_grad_by_norm_(self, grad, max_norm):
    #     """
    #     in-place gradient clipping.
    #     :param grad: list of gradients
    #     :param max_norm: maximum norm allowable
    #     :return:
    #     """
    #
    #     total_norm = 0
    #     counter = 0
    #     for g in grad:
    #         param_norm = g.data.norm(2)
    #         total_norm += param_norm.item() ** 2
    #         counter += 1
    #     total_norm = total_norm ** (1. / 2)
    #
    #     clip_coef = max_norm / (total_norm + 1e-6)
    #     if clip_coef < 1:
    #         for g in grad:
    #             g.data.mul_(clip_coef)
    #
    #     return total_norm / counter

    def clip_grad_by_norm_(self, grad, max_norm):
        for g in grad:
            if g is None:
                continue
            g.data.clamp_(-max_norm, max_norm)
        return grad

    def set_zero_grad(self, grad, begin, end):
        i = 0
        for g in grad:
            if g is None:
                continue
            if i >= begin and i < end:
                d = g.data.new(self.g.size())
                d.fill_(0)
                g.data = d
            i+=1
        return grad

    def forward(self, xs, ys, xq, yq):
        task_num, _, _ = xs.size()
        # losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        losses_q = [0, 0]
        for i in range(task_num):
            logits = self.net.pretrain(xs[i], vars=None)
            loss = F.mse_loss(logits, ys[i])
            grad = torch.autograd.grad(loss, self.net.parameters()[:6])
            grad = self.clip_grad_by_norm_(grad, 10)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()[:6])))
            fast_weights.extend(list(self.net.parameters()[6:]))
            for k in range(1, self.update_step):
                logits = self.net.pretrain(xs[i], vars=fast_weights)
                loss = F.mse_loss(logits, ys[i])
                grad = torch.autograd.grad(loss, fast_weights[:6])
                grad = self.clip_grad_by_norm_(grad, 10)
                fast_weights_t = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights[:6])))
                fast_weights_t.extend(fast_weights[6:])
                fast_weights = fast_weights_t
            for k in range(0, self.update_step):
                logits = self.net.prune(xs[i], vars=fast_weights)
                loss = F.mse_loss(logits, ys[i])
                # print(type(self.net.bim))
                #loss.backward()
                grad = torch.autograd.grad(loss, self.net.bim)
                # grad = torch.autograd.grad(loss, fast_weights[6:])
                print(grad)
                grad = self.clip_grad_by_norm_(grad, 10)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            for k in range(0, self.update_step):
                logits = self.net.prune(xs[i], vars=fast_weights)
                loss = F.mse_loss(logits, ys[i])
                grad = torch.autograd.grad(loss, fast_weights)
                grad = self.clip_grad_by_norm_(grad, 10)
                grad = self.set_zero_grad(grad, 6, 9)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # with no grad
            logits_q = self.net.prune(xq[i], self.net.parameters())
            loss_q = F.mse_loss(logits_q, yq[i])
            losses_q[0] += loss_q

            logits_q = self.net(xq[i], fast_weights)
            loss_q = F.mse_loss(logits_q, yq[i])
            losses_q[1] += loss_q

        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        # nn.utils.clip_grad_value_(self.net.vars, clip_value=1)
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        return losses_q

    def finetunning(self, xs, ys, xq, yq):
        losses = [0, 0]
        losses_q = [0, 0]
        net = deepcopy(self.net)
        logits = self.net.pretrain(xs, vars=None)
        loss = F.mse_loss(logits, ys)
        losses[0] += loss
        grad = torch.autograd.grad(loss, self.net.parameters())
        grad = self.clip_grad_by_norm_(grad, 10)
        grad = self.set_zero_grad(grad, 6, 9)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
        for k in range(1, self.updata_step):
            logits = self.net.pretrain(xs, vars=fast_weights)
            loss = F.mse_loss(logits, ys[i])
            grad = torch.autograd.grad(loss, fast_weights)
            grad = self.clip_grad_by_norm_(grad, 10)
            grad = self.set_zero_grad(grad, 6, 9)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        for k in range(0, self.update_step):
            logits = self.net.prune(xs, vars=fast_weights)
            loss = F.mse_loss(logits, ys)
            grad = torch.autograd.grad(loss, fast_weights)
            grad = self.clip_grad_by_norm_(grad, 10)
            grad = self.set_zero_grad(grad, 0, 6)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        for k in range(0, self.update_step):
            logits = self.net.prune(xs, vars=fast_weights)
            loss = F.mse_loss(logits, ys)
            grad = torch.autograd.grad(loss, fast_weights)
            grad = self.clip_grad_by_norm_(grad, 10)
            grad = self.set_zero_grad(grad, 6, 9)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            loss[1] += loss

            logits_q = self.net.prune(xq, self.net.parameters())
            loss_q = F.mse_loss(logits_q, yq)
            losses_q[0] += loss_q

            logits_q = self.net(xq, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, yq)
            losses_q[1] += loss_q

        del net

        return losses, losses_q, logits_q


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=100)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    maml = Meta(args)
