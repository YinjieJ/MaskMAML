import  torch
from    torch import nn
from    torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import  numpy as np
from copy import deepcopy


# requires_grad=True
MASK_SCALE = 1e-2
DEFAULT_THRESHOLD = 5e-3

class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class Prune(nn.Module):
    def __init__(self):
        super(Prune, self).__init__()
        self.vars = nn.ParameterList()
        w0 = nn.Parameter(torch.ones([40, 1]))
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0)
        b0 = nn.Parameter(torch.zeros([40]))
        self.vars.append(b0)
        w1 = nn.Parameter(torch.ones([40, 40]))
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w1)
        b1 = nn.Parameter(torch.zeros([40]))
        self.vars.append(b1)
        w2 = nn.Parameter(torch.ones([1, 40]))
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        b2 = nn.Parameter(torch.zeros([1]))
        self.vars.append(b2)
        mw0 = nn.Parameter(torch.Tensor(1).fill_(MASK_SCALE))
        self.vars.append(mw0)
        mw1 = nn.Parameter(torch.Tensor(40).fill_(MASK_SCALE))
        self.vars.append(mw1)
        mw2 = nn.Parameter(torch.Tensor(40).fill_(MASK_SCALE))
        self.vars.append(mw2)
        self.threshold_fn = Binarizer()
        self.threshold = DEFAULT_THRESHOLD

    def forward(self, *input):
        pass


    def pretrain(self, x, vars=None):
        if vars is None:
            vars = self.vars[:6]

        idx = 0
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        x = F.leaky_relu(x)
        idx += 2
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        x = F.leaky_relu(x)
        idx += 2
        w, b = vars[idx], vars[idx + 1]
        x = F.linear(x, w, b)
        idx += 2

        assert idx == len(vars)

        return x

    def prune(self, x, vars=None):
        if vars is None:
            vars = self.vars

        idx = 0
        w, b = vars[idx], vars[idx+1]
        realm = vars[idx//2+6]
        bim = self.threshold_fn.apply(realm, self.threshold)
        # bim = bim.repeat(w.shape[0],1)
        wt = w.mul(bim)
        x = F.linear(x, wt, b)
        x = F.leaky_relu(x)
        idx += 2
        w, b = vars[idx], vars[idx + 1]
        realm = vars[idx//2 + 6]
        bim = self.threshold_fn.apply(realm, self.threshold)
        # bim = bim.repeat(w.shape[0],1)
        wt = w.mul(bim)
        x = F.linear(x, wt, b)
        x = F.leaky_relu(x)
        idx += 2
        w, b = vars[idx], vars[idx + 1]
        realm = vars[idx // 2 + 6]
        bim = self.threshold_fn.apply(realm, self.threshold)
        # bim = bim.expand(w.shape[0], -1)
        # bim.register_hook(print)
        # bim = bim.reshape(1,-1)
        bim = bim
        wt = bim.mul(w)
        x = F.linear(x, wt, b)
        idx += 2

        assert idx/2 == len(vars)/3

        return x

    def get_cnum(self, vars=None):
        if vars is None:
            vars = self.vars

        c0 = int(sum((self.threshold_fn.apply(vars[6], self.threshold) == 1).int()).cpu().numpy())
        c1 = int(sum((self.threshold_fn.apply(vars[7], self.threshold) == 1).int()).cpu().numpy())
        c2 = int(sum((self.threshold_fn.apply(vars[8], self.threshold) == 1).int()).cpu().numpy())

        return [c0, c1, c2]


    def parameters(self):
        return self.vars



if __name__=='__main__':
    # pr = Prune()
    # x = np.array([[1.], [2.], [3.]], dtype=np.float32)
    # pr.pretrain(torch.from_numpy(x))
    # pr.prune(torch.from_numpy(x))
    # print(pr.get_cnum(pr.parameters()))
    # print(pr.parameters()[1:3].extend(pr.parameters()[3:5]))
    x = torch.Tensor([1, 0, 1])
    w0 = nn.Parameter(torch.ones([40, 3]))
    r = x * w0
    print(x.shape)
    print(r)