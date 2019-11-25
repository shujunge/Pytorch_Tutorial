import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TimeDistributed(nn.Module):
  
  def __init__(self, module, batch_first=True):
    super(TimeDistributed, self).__init__()
    self.module = module
    self.batch_first = batch_first
  
  def forward(self, x):
    
    if len(x.size()) <= 2:
      return self.module(x)
    
    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
    
    y = self.module(x_reshape)
    
    # We have to reshape Y
    if self.batch_first:
      y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
    else:
      y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
    
    return y


class LSTM_Attention(nn.Module):
  
  def __init__(self, inputDim, hiddenNum, outputDim, layerNum, seq_len,hidden_size, merge="concate"):
    
    super(LSTM_Attention, self).__init__()
    self.hiddenNum = hiddenNum
    self.merge = merge
    self.seq_len = seq_len
    self.layerNum = layerNum
    self.inputDim = inputDim
    self.hidden_size = hidden_size
    self.att_fc = nn.Linear(hiddenNum, 1)
    self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hiddenNum,
                        num_layers=self.layerNum, dropout=0.3,
                        batch_first=True, )
    
    self.time_distribut_layer = TimeDistributed(self.att_fc)
    if merge == "mean":
      self.dense = nn.Linear(hiddenNum, outputDim)
    if merge == "concate":
      self.dense = nn.Linear(hiddenNum * seq_len, self.hidden_size)
      self.dense2 = nn.Linear(self.hidden_size, outputDim)
  
  def forward(self, x, batchSize):
    
    h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)).to(device)
    c0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum)).to(device)
    rnnOutput, hn = self.cell(x, (h0, c0))
    
    attention_out = self.time_distribut_layer(rnnOutput)
    attention_out = attention_out.view((batchSize, -1))
    attention_out = F.softmax(attention_out)
    attention_out = attention_out.view(batchSize, -1, 1)
    
    rnnOutput = rnnOutput * attention_out
    
    if self.merge == "mean":
      sum_hidden = torch.mean(rnnOutput, 1)
      x = sum_hidden.view(-1, self.hiddenNum)
    if self.merge == "concate":
      rnnOutput = rnnOutput.contiguous()
      x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)
    
    x = self.dense(x)
    x = nn.ReLU()(x)
    fcOutput = self.dense2(x)
    
    return fcOutput


class GRU_Attention(nn.Module):
  
  def __init__(self, inputDim, hiddenNum, outputDim, layerNum, seq_len,hidden_size, merge="concate"):
    
    super(GRU_Attention, self).__init__()
    self.hiddenNum = hiddenNum
    self.merge = merge
    self.seq_len = seq_len
    self.layerNum = layerNum
    self.inputDim = inputDim
    self.hidden_size = hidden_size
    self.att_fc = nn.Linear(hiddenNum, 1)
    self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                       num_layers=self.layerNum, dropout=0.3,
                       batch_first=True, )
    
    self.time_distribut_layer = TimeDistributed(self.att_fc)
    if merge == "mean":
      self.dense = nn.Linear(hiddenNum, outputDim)
    if merge == "concate":
      self.dense = nn.Linear(hiddenNum * seq_len, self.hidden_size)
      self.dense2 = nn.Linear(self.hidden_size, outputDim)
  
  def forward(self, x, batch_size):
    
    h0 = Variable(torch.zeros(self.layerNum * 1, batch_size, self.hiddenNum)).to(device)
    rnnOutput, hn = self.cell(x, h0)
    
    attention_out = self.time_distribut_layer(rnnOutput)
    attention_out = attention_out.view((batch_size, -1))
    attention_out = F.softmax(attention_out)
    attention_out = attention_out.view(batch_size, -1, 1)
    
    rnnOutput = rnnOutput * attention_out
    
    if self.merge == "mean":
      sum_hidden = torch.mean(rnnOutput, 1)
      x = sum_hidden.view(-1, self.hiddenNum)
    if self.merge == "concate":
      rnnOutput = rnnOutput.contiguous()
      x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)
    
    x = self.dense(x)
    x = nn.ReLU()(x)
    fcOutput = self.dense2(x)
    
    return fcOutput

