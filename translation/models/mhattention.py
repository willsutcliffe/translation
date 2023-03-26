import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset


class MultiHeadAttention(nn.Module):
    """
    Class for MultiHeadAttention

    Attributes
    ----------

    d_key: int
        dimension of key

    n_heads: int
        number of heads

    key: nn.Linear
        linear transformation for the key, W_k x

    query: nn.Linear
        linear transformation for the query, W_q x

    value: nn.Linear
        linear transformation for the value, W_v x

    fc: nn.Linear
        final feed forward layer

    casual: bool
        take into account casuality or not, required for
        the decoded.



    """
    def __init__(self, d_key, d_model, n_heads, max_len, causal=False):
        """
        Constructor for Multi-Head-Attention

        :param d_key: int
            dimension of key
        :param d_model: int
            dimension of the model
        :param n_heads: int
            number of heads
        :param max_len: int
            max length of sequence
        :param causal: bool
            True required for decoder
        """
        super().__init__()

        self.d_key = d_key
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_key * n_heads)
        self.query = nn.Linear(d_model, d_key * n_heads)
        self.value = nn.Linear(d_model, d_key * n_heads)

        self.fc = nn.Linear(d_key * n_heads, d_model)

        self.causal = causal

        if causal:
            cm = torch.tril(torch.ones(max_len, max_len))
            self.register_buffer(
                "causal_mask",
                cm.view(1, 1, max_len, max_len)
            )
        # store attention weights
        self.att_weights=None

    def forward(self, q, k, v, pad_mask=None):
        """
        Forward propagation for Multi-Head-Attention
        :param q: torch.Tensor
        :param k: torch.Tensor
        :param v: torch.Tensor
        :param pad_mask: torch.Tensor
        :return: torch.Tensor
        """
        # N x T x (h d_key)
        q = self.query(q)
        # N x T x (h d_key)
        k = self.key(k)
        # N x T x (h d_v)
        v = self.value(v)

        N = q.shape[0]
        T_out = q.shape[1]
        T_in = k.shape[1]

        # (N, T, h, d_key) -> (N, h, T, d_key)
        q = q.view(N, T_out, self.n_heads, self.d_key).transpose(1, 2)
        k = k.view(N, T_in, self.n_heads, self.d_key).transpose(1, 2)
        v = v.view(N, T_in, self.n_heads, self.d_key).transpose(1, 2)

        # compute attention
        # (N, h, T, d_key) x (N, h, d_key, T) -> (N, h, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_key)
        if pad_mask is not None:
            attn_scores = attn_scores.masked_fill(
                pad_mask[:, None, None, :] == 0, float('-inf'))
        if self.causal:
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:, :, :T_out, :T_in] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        self.att_weights = attn_weights

        # (N, h, T, T) x (N, h, T, d_key) --> (N, h, T, d_key)
        A = attn_weights @ v

        # (N, T, h, d_key)
        A = A.transpose(1, 2)
        # (N, T, h*d_key)
        A = A.contiguous().view(N, T_out, self.d_key * self.n_heads)

        return self.fc(A)
