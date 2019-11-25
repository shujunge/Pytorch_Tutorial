import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

#softmax
class MyCrossEntropyLoss(nn.Module):
  def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
    super(MyCrossEntropyLoss, self).__init__()
    self.weight = weight
    self.reduction = reduction
    self.ignore_index = ignore_index
  
  def forward(self, input, target):
    return F.cross_entropy(input, target, weight=self.weight,
                           ignore_index=self.ignore_index, reduction=self.reduction)


# SphereFace
class SphereProduct(nn.Module):
  r"""Implement of large margin cosine distance: :
  Args:
      in_features: size of each input sample
      out_features: size of each output sample
      m: margin
      cos(m*theta)
  """
  
  def __init__(self, in_features, out_features, m=4):
    super(SphereProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.m = m
    self.base = 1000.0
    self.gamma = 0.12
    self.power = 1
    self.LambdaMin = 5.0
    self.iter = 0
    self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform(self.weight)
    
    # duplication formula
    # 将x\in[-1,1]范围的重复index次映射到y\[-1,1]上
    self.mlambda = [
      lambda x: x ** 0,
      lambda x: x ** 1,
      lambda x: 2 * x ** 2 - 1,
      lambda x: 4 * x ** 3 - 3 * x,
      lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
      lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
    ]
    """
    执行以下代码直观了解mlambda
    import matplotlib.pyplot as  plt

    mlambda = [
        lambda x: x ** 0,
        lambda x: x ** 1,
        lambda x: 2 * x ** 2 - 1,
        lambda x: 4 * x ** 3 - 3 * x,
        lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
        lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
    ]
    x = [0.01 * i for i in range(-100, 101)]
    print(x)
    for f in mlambda:
        plt.plot(x,[f(i) for i in x])
        plt.show()
    """
  
  def forward(self, input, label):
    # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
    self.iter += 1
    self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
    
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
    cos_theta = cos_theta.clamp(-1, 1)
    cos_m_theta = self.mlambda[self.m](cos_theta)
    theta = cos_theta.data.acos()
    k = (self.m * theta / 3.14159265).floor()
    phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
    NormOfFeature = torch.norm(input, 2, 1)
    
    # --------------------------- convert label to one-hot ---------------------------
    one_hot = torch.zeros(cos_theta.size())
    one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
    one_hot.scatter_(1, label.view(-1, 1), 1)
    
    # --------------------------- Calculate output ---------------------------
    output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
    output *= NormOfFeature.view(-1, 1)
    
    return output
  
  def __repr__(self):
    return self.__class__.__name__ + '(' \
           + 'in_features=' + str(self.in_features) \
           + ', out_features=' + str(self.out_features) \
           + ', m=' + str(self.m) + ')'


##CosFace
class AddMarginProduct(nn.Module):
  r"""Implement of large margin cosine distance: :
  Args:
      in_features: size of each input sample
      out_features: size of each output sample
      s: norm of input feature
      m: margin
      cos(theta) - m
  """

  def __init__(self, in_features, out_features, s=30.0, m=0.40):
    super(AddMarginProduct, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.s = s
    self.m = m
    self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
    nn.init.xavier_uniform_(self.weight)

  def forward(self, input, label):
    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(input), F.normalize(self.weight))
    phi = cosine - self.m
    # --------------------------- convert label to one-hot ---------------------------
    one_hot = torch.zeros(cosine.size(), device='cuda')
    # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    # you can use torch.where if your torch.__version__ is 0.4
    output *= self.s
    # print(output)
  
    return output

  def __repr__(self):
    return self.__class__.__name__ + '(' \
           + 'in_features=' + str(self.in_features) \
           + ', out_features=' + str(self.out_features) \
           + ', s=' + str(self.s) \
           + ', m=' + str(self.m) + ')'
  

# ArcFace
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Parameter 的用途：
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的
        # https://www.jianshu.com/p/d8b77cc02410
        # 初始化权重
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

# FocalLoss
class FocalLoss(nn.Module):
  r"""
      This criterion is a implemenation of Focal Loss, which is proposed in
      Focal Loss for Dense Object Detection.

          Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

      The losses are averaged across observations for each minibatch.

      Args:
          alpha(1D Tensor, Variable) : the scalar factor for this criterion
          gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                 putting more focus on hard, misclassiﬁed examples
          size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                              However, if the field size_average is set to False, the losses are
                              instead summed for each minibatch.

  """
  
  def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
    super(FocalLoss, self).__init__()
    if alpha is None:
      self.alpha = Variable(torch.ones(class_num, 1))
    else:
      if isinstance(alpha, Variable):
        self.alpha = alpha
      else:
        self.alpha = Variable(alpha)
    self.gamma = gamma
    self.class_num = class_num
    self.size_average = size_average
  
  def forward(self, inputs, targets):
    N = inputs.size(0)
    print(N)
    C = inputs.size(1)
    P = F.softmax(inputs)
    
    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    # print(class_mask)
    
    
    if inputs.is_cuda and not self.alpha.is_cuda:
      self.alpha = self.alpha.cuda()
    alpha = self.alpha[ids.data.view(-1)]
    
    probs = (P * class_mask).sum(1).view(-1, 1)
    
    log_p = probs.log()
    # print('probs size= {}'.format(probs.size()))
    # print(probs)
    
    batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
    # print('-----bacth_loss------')
    # print(batch_loss)
    
    
    if self.size_average:
      loss = batch_loss.mean()
    else:
      loss = batch_loss.sum()
    return loss


# alpha = torch.rand(21, 1)
# print(alpha)
# FL = FocalLoss(class_num=5, gamma=0)
# CE = nn.CrossEntropyLoss()
# N = 4
# C = 5
# inputs = torch.rand(N, C)
# targets = torch.LongTensor(N).random_(C)
# inputs_fl = Variable(inputs.clone(), requires_grad=True)
# targets_fl = Variable(targets.clone())
#
# inputs_ce = Variable(inputs.clone(), requires_grad=True)
# targets_ce = Variable(targets.clone())
# print('----inputs----')
# print(inputs)
# print('---target-----')
# print(targets)
#
# fl_loss = FL(inputs_fl, targets_fl)
# ce_loss = CE(inputs_ce, targets_ce)
# print('ce = {}, fl ={}'.format(ce_loss.item(), fl_loss.item()))
# fl_loss.backward()
# ce_loss.backward()
#
# print(inputs_fl.grad.data)
# print(inputs_ce.grad.data)