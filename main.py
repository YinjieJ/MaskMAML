import  torch, os
import  numpy as np
from    data import Gen
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta import Meta

class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    print(args)
    device = torch.device('cuda')
    maml = Meta(args).to(device)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    trainset = Gen(args.task_num, args.k_spt, args.k_qry)
    testset = Gen(args.task_num, args.k_spt, args.k_qry*10)

    for epoch in range(args.epoch):
        ind = [i for i in range(trainset.xs.shape[0])]
        np.random.shuffle(ind)
        xs, ys = torch.Tensor(trainset.xs[ind]).to(device), torch.Tensor(trainset.ys[ind]).to(device)
        xq, yq = torch.Tensor(trainset.xq[ind]).to(device),torch.Tensor(trainset.yq[ind]).to(device)
        maml.train()
        loss = maml(xs, ys, xq, yq, epoch)
        print('Epoch: {} Initial loss: {} Train loss: {}'.format(epoch, loss[0]/args.task_num, loss[-1]/args.task_num))
        if (epoch+1) % 50 == 0:
            print("Evaling the model...")
            torch.save(maml.state_dict(), 'save.pt')
            # del(maml)
            # maml = Meta(args).to(device)
            # maml.load_state_dict(torch.load('save.pt'))
            maml.eval()
            i = random.randint(0, testset.xs.shape[0]-1)
            xs, ys = torch.Tensor(testset.xs[i]).to(device), torch.Tensor(testset.ys[i]).to(device)
            xq, yq = torch.Tensor(testset.xq[i]).to(device), torch.Tensor(testset.yq[i]).to(device)
            losses, losses_q, logits_q, _ = maml.finetunning(xs, ys, xq, yq)
            print('Epoch: {} Initial loss: {} Test loss: {}'.format(epoch, losses_q[0], losses_q[-1]))

if __name__ == '__main__':
    sys.stdout = Logger("out.txt", sys.stdout)
    # sys.stderr = Logger("out.txt", sys.stderr)       # redirect std err, if necessary
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1000)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--warm_step', type=int, help='warm steps before train masks', default=1000)

    args = argparser.parse_args()
    main()