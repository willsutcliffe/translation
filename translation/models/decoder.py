from translation.models.mhattention import MultiHeadAttention
from translation.models.positional import PositionalEncoding
import torch.nn as nn
import torch
import math

class DecoderBlock(nn.Module):
    def __init__(self, d_key, d_model, n_heads, max_len, dropout_prob=0.1):
        """

        :param d_key:
        :param d_model:
        :param n_heads:
        :param max_len:
        :param dropout_prob:
        """
        super().__init__()

        self.linear1 = nn.LayerNorm(d_model)
        self.linear2 = nn.LayerNorm(d_model)
        self.linear3 = nn.LayerNorm(d_model)
        self.mhatt1 = MultiHeadAttention(d_key, d_model, n_heads, max_len, causal=True)
        self.mhatt2 = MultiHeadAttention(d_key, d_model, n_heads, max_len, causal=False)
        self.nn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        """

        :param enc_output:
        :param dec_input:
        :param enc_mask:
        :param dec_mask:
        :return:
        """
        # self-attention on decoder input
        x = self.linear1(
            dec_input + self.mhatt1(dec_input, dec_input, dec_input, dec_mask))
        # multi-head attention including output of the encoder
        x = self.linear2(x + self.mhatt2(x, enc_output, enc_output, enc_mask))
        x = self.linear3(x + self.nn(x))
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 d_key,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout_prob):
        """

        :param vocab_size:
        :param max_len:
        :param d_key:
        :param d_model:
        :param n_heads:
        :param n_layers:
        :param dropout_prob:
        """
        super().__init__()


        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)

        transformer_blocks = [
            DecoderBlock(
                d_key,
                d_model,
                n_heads,
                max_len,
                dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.linear = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        """

        :param enc_output:
        :param dec_input:
        :param enc_mask:
        :param dec_mask:
        :return:
        """
        x = self.embedding(dec_input)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(enc_output, x, enc_mask, dec_mask)
        x = self.linear(x)
        x = self.fc(x)
        return x