import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


alpha = torch.Tensor(torch.randn(2, 1))  ##torch.float32
y =alpha.permute(1,0).contiguous()
print(y)
alpha.add_(1)
print(y)
alpha = torch.Tensor([0.001])
print(alpha.item())












CE = nn.CrossEntropyLoss()
N = 4
C = 5
inputs = torch.rand(N, C)
targets = torch.LongTensor(N).random_(C)
inputs_fl = Variable(inputs.clone(), requires_grad=True)
targets_fl = Variable(targets.clone())

inputs_ce = Variable(inputs.clone(), requires_grad=True)
targets_ce = Variable(targets.clone())
print('----inputs----')
print(inputs.dtype) #torch.float32
print('---target-----')
print(targets.dtype) #torch.int64


ce_loss = CE(inputs_ce, targets_ce)
print('ce = {}'.format(ce_loss.item()))

ce_loss.backward()

print(inputs_ce.grad.data)

# m = nn.Sigmoid()
loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)

print('----inputs----')
print(input.dtype) #torch.float32
print('---target-----')
print(target.dtype) #torch.float32

output = loss(input, target)
output.backward()


"""
summary:

contiguous 为深拷贝

CrossEntropyLoss =logsofmax + NllLoss
inputs.dtype: torch.float32
targets.dtype: torch.int64


BCEWithLogitsLoss = Logits + BCELoss
inputs.dtype: torch.float32
targets.dtype: torch.float32

"""


