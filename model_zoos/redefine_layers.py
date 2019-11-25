import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    # nn.Module.__init__(self)
    super(MyLinear, self).__init__()
    self.w = nn.Parameter(torch.randn(out_features, in_features))  # nn.Parameter是特殊Variable
    self.b = nn.Parameter(torch.randn(out_features))
  
  def forward(self, x):
    # wx+b
    return F.linear(x, self.w, self.b)


class Net(nn.Module):
  def __init__(self,classes=10):
    super(Net, self).__init__()
    self.outdim= classes
    self.layer1 = nn.Sequential(nn.Conv2d(3,  16, 3), nn.BatchNorm2d(16), nn.ReLU())
    self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.BatchNorm2d(32), nn.ReLU())
    self.layer3 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU())
  def forward(self,x):
    out = self.layer1(x)
    out =self.layer2(out)
    out= self.layer3(out)
    out =out.reshape(out.size(0),-1)
    out = nn.Linear(out.size(1),self.outdim)(out)
    return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class MyCrossEntropyLoss(nn.Module):

  def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
    super(MyCrossEntropyLoss, self).__init__()
    self.weight = weight
    self.reduction = reduction
    self.ignore_index = ignore_index
  

  def forward(self, input, target):
    return F.cross_entropy(input, target, weight=self.weight,
                           ignore_index=self.ignore_index, reduction=self.reduction)


class MyLoss(nn.Module):
  def __init__(self):
    super(MyLoss, self).__init__()
    print('1')

  def forward(self, pred, truth):
    truth = torch.mean(truth, 1)
    truth = truth.view(-1, 2048)
    pred = pred.view(-1, 2048)
    return torch.mean(torch.mean((pred - truth) ** 2, 1), 0)






