import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2048, dropout_prob=0.1):

    super().__init__()
    self.dropout = nn.Dropout(p=dropout_prob)
    position = torch.arange(max_len).unsqueeze(1)
    exp_term = torch.arange(0, d_model, 2)
    div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe)

  def forward(self, x):
    #shape of x: N x T x D
    x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)