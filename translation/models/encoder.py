from translation.models.mhattention import MultiHeadAttention
from translation.models.positional import PositionalEncoding
import torch.nn as nn
import torch
import math


class EncoderBlock(nn.Module):
    """
    Class encapsulating block of transformer Encoder
    """
    def __init__(self, d_key, d_model, n_heads, max_len, dropout_prob=0.1):
        """
        Constructor for Encoder

        :param d_key: int
        :param d_model: int
        :param n_heads: int
        :param max_len: int
        :param dropout_prob: float
        """
        super().__init__()

        self.linear1 = nn.LayerNorm(d_model)
        self.linear2 = nn.LayerNorm(d_model)
        self.mhatt = MultiHeadAttention(d_key, d_model, n_heads, max_len, causal=False)
        self.nn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        

    def forward(self, x, pad_mask=None):
        """

        :param x:
        :param pad_mask:
        :return:
        """
        x = self.linear1(x + self.mhatt(x, x, x, pad_mask))
        x = self.linear2(x + self.nn(x))
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    Class encapsulating Encoder of Transformer
    """
    def __init__(self,
                 vocab_size,
                 max_len,
                 d_key,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout_prob,
                 n_classes = None):
        """

        :param vocab_size:
        :param max_len:
        :param d_key:
        :param d_model:
        :param n_heads:
        :param n_layers:
        :param dropout_prob:
        :param n_classes:
        """

        super().__init__()
        # Define embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # define positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)

        # Transformer
        transformer_blocks = [
            EncoderBlock(
                d_key,
                d_model,
                n_heads,
                max_len,
                dropout_prob) for _ in range(n_layers)]

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.linear = nn.LayerNorm(d_model)
        if n_classes != None and type(n_classes) == int:
            print("Using encoder with a single target.")
            self.fc = nn.Linear(d_model, n_classes)
            self.final = self.many_to_one
        elif n_classes != None and type(n_classes) != int:
            print(f"Wrong parameter type for n_classes. Expected int and not {type(n_classes)}")
            raise TypeError
        else:
            print("Using encoder without a target.")
            self.final = self.encoded_output

    def encoded_output(self, x):
        return  self.linear(x)

    def many_to_one(self, x):
        x = x[:, 0, :]
        return self.fc(x)

    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, pad_mask)

        self.final(x)
        return x

