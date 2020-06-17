import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import math

import models
import masks

# DEFAULT = 'VGG16'

class MaskModel(nn.Module):
    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None, 
                 make_model=True, phase=0):
        super(MaskModel, self).__init__()
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, 0)
    
#     def add_cfg(self, cfg, name='new'):
#             self.cfg[name] = cfg
    
#     def get_cfg(self):
#         cfg = self.cfg[DEFAULT]
#         i = -1
#         k = 1
#         for m in self.model.modules():
#             if 'Wise' in str(type(m)):
#                 if i != -1:
#                     cfg[i] = int(sum((m.threshold_fn(m.mask_real) == 1).int()).cpu().numpy())
#                 i += k
#                 k = 1
#             if 'MaxPool2d' in str(type(m)):
#                 k += 1
#         print(cfg)
#         self.add_cfg(cfg, 'new')
            
    
    def make_model(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', threshold=None, phase=0):
        '''creates a model including original model and masked model'''
        if phase == 0:
            self.pre = models.ResNet101()
            print('Creating pre-train model: No mask layers.')
        elif phase == 2:
            self.post = models.EfficientNetB0('new', self.cfg)
            mod = list(zip(self.post.modules(), self.model.modules()))[::-1]
            tm = None
            for mmp, mm in mod:
                if 'Wise' in str(type(mm)):
                    if tm:
                        data = mm.weight.data[tm.threshold_fn(tm.mask_real) == 1,...]
                        data = data[:,mm.threshold_fn(mm.mask_real) == 1,...]
                        mmp.weight.data.copy_(data)
                        mmp.bias.data.copy_(mm.bias.data[tm.threshold_fn(tm.mask_real) == 1])
                    else:
                        print(mmp.weight.size())
                        print((mm.threshold_fn(mm.mask_real) == 1).size())
                        mmp.weight.data.copy_(mm.weight.data[:,mm.threshold_fn(mm.mask_real) == 1,...])
                        mmp.bias.data.copy_(mm.bias.data)
                    tm = mm
            print('Creating post-train model: No mask layers')
        else:
            self.model = masks.ResNet101(mask_init, mask_scale, threshold_fn, elementwise = False)
            for m, mp in zip(self.model.modules(), self.pre.modules()):
                if 'Wise' in str(type(m)):
                    m.weight.data.copy_(mp.weight.data)
                    if hasattr(m.bias, 'data'):
                        m.bias.data.copy_(mp.bias.data)
            print('Creating model: Mask layers created.')
        
    def forward(self, x, phase=0):
        if phase == 0:
            return(self.pre(x))
        elif phase == 2:
            return(self.post(x))
        else:
            return(self.model(x))
    
def test():
    x = torch.randn(2,3,32,32)
    net = MaskModel(phase = 0)
    y = net(x, 0)
    print(y.size())
    net.make_model(phase = 1)
    y = net(x, 1)
    print(y.size())
#     net.get_cfg()
#     net.make_model(phase = 2)
#     y = net(x, 2)
#     print(y.size())

# test()