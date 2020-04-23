import os
import random
import math
import numpy as np
from scipy.special import logsumexp

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import ModuleList
from torchvision.utils import make_grid

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_runtime(seed=0, cuda_dev_id=[0]):
    """Initialize CUDA, CuDNN and the random seeds."""
    # Setup CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(cuda_dev_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
        for i in cuda_dev_id[1:]:
            os.environ["CUDA_VISIBLE_DEVICES"] += "," + str(i)

    # global cuda_dev_id
    _cuda_device_id = cuda_dev_id
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False 
    # Fix random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TotalAverage():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.mass = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, val, mass=1):
        self.val = val
        self.mass += mass
        self.sum += val * mass
        self.avg = self.sum / self.mass


class MovingAverage():
    def __init__(self, intertia=0.9):
        self.intertia = intertia
        self.reset()

    def reset(self):
        self.avg = 0.

    def update(self, val):
        self.avg = self.intertia * self.avg + (1 - self.intertia) * val


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def write_conv(writer, model, epoch, sobel=False):
    if not sobel:
        conv1_ = make_grid(list(ModuleList(list(model.children())[0].children())[0].parameters())[0],
                           nrow=8, normalize=True, scale_each=True)
        writer.add_image('conv1', conv1_, epoch)
    else:
        conv1_sobel_w = list(ModuleList(list(model.children())[0].children())[0].parameters())[0]
        conv1_ = make_grid(conv1_sobel_w[:, 0:1, :, :], nrow=8,
                           normalize=True, scale_each=True)
        self.writer.add_image('conv1_sobel_1', conv1_, epoch)
        conv2_ = make_grid(conv1_sobel_w[:, 1:2, :, :], nrow=8,
                           normalize=True, scale_each=True)
        self.writer.add_image('conv1_sobel_2', conv2_, epoch)
        conv1_x = make_grid(torch.sum(conv1_sobel_w[:, :, :, :], 1, keepdim=True), nrow=8,
                            normalize=True, scale_each=True)
        writer.add_image('conv1', conv1_x, epoch)


### LP stuff ###
def absorb_bn(module, bn_module):
    w = module.weight.data
    if module.bias is None:
        if isinstance(module, nn.Linear):
            zeros = torch.Tensor(module.out_features).zero_().type(w.type())
        else:
            zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    b = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    if isinstance(module, nn.Conv2d):
        w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
    else:
        w.mul_(invstd.unsqueeze(1).expand_as(w))
    b.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine:
        if isinstance(module, nn.Conv2d):
            w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
        else:
            w.mul_(bn_module.weight.data.unsqueeze(1).expand_as(w))
        b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    bn_module.reset_parameters()
    bn_module.register_buffer('running_mean', None)
    bn_module.register_buffer('running_var', None)
    bn_module.affine = False
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)


def search_absorb_bn(model):
    prev = None
    for m in model.children():
        if is_bn(m) and is_absorbing(prev):
            print("absorbing",m)
            absorb_bn(prev, m)
        search_absorb_bn(m)
        prev = m


class View(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


def sequential_skipping_bn_cut(model):
    mods = []
    layers = list(model.features) + [View()]
    if 'sobel' in dict(model.named_children()).keys():
        layers = list(model.sobel) + layers
    for m in nn.Sequential(*(layers)).children():
        if not is_bn(m):
            mods.append(m)
    return nn.Sequential(*mods)


def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def warmup_batchnorm(model, data_loader, device, batches=100):
    """
    Run some batches through all parts of the model to warmup the running
    stats for batchnorm layers.
    """
    model.train()
    for i, q in enumerate(data_loader):
        images = q[0]
        if i == batches:
            break
        images = images.to(device)
        _ = model(images)

def init_pytorch_defaults(m, version='041'):
    '''
    copied from AMDIM repo: https://github.com/Philip-Bachman/amdim-public/
    note from me: haven't checked systematically if this improves results
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)


def search_set_bn_eval(model,toeval):
    for m in model.children():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if toeval:
                m.eval()
            else:
                m.train()
        search_set_bn_eval(m, toeval)

def prepmodel(model, modelpath):
    dat = torch.load(modelpath, map_location=lambda storage, loc: storage)  # ['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in dat.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    del dat
    for param in model.features.parameters():
        param.requires_grad = False

    if model.headcount > 1:
        for i in range(model.headcount):
            setattr(model, "top_layer%d" % i, None)

    model.top_layer = nn.Sequential(nn.Linear(2048, 1000))
    model.headcount = 1
    model.withfeature = False
    model.return_feature_only = False