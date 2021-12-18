import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.errors import DimensionError


class DecoderLayer(nn.Module):
    def __init__(
            self, n_embed, n_head, dim_ff, attn_dropout, residual_dropout):
        super().__init__()
        self.norm1 = LayerNormalizer(n_embed)
        self.masked_attn = MultiHeadAttention(
            n_head, n_embed, attn_dropout, residual_dropout
        )
        self.norm2 = LayerNormalizer(n_embed)
        self.attn = MultiHeadAttention(
            n_head, n_embed, attn_dropout, residual_dropout
        )
        self.norm3 = LayerNormalizer(n_embed)
        self.ff = FeedForward(n_embed, dim_ff, residual_dropout)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        masked_attn_input = self.norm1(x)
        x = x + self.masked_attn(
            masked_attn_input,
            masked_attn_input,
            masked_attn_input,
            decoder_mask
        )

        attn_input = self.norm2(x)
        x = x + self.attn(
            attn_input,
            encoder_output,
            encoder_output,
            encoder_mask
        )

        ff_input = self.norm3(x)
        x = x + self.ff(ff_input)

        return x


class EncoderLayer(nn.Module):
    def __init__(
            self, n_embed, n_head, dim_ff, attn_dropout, residual_dropout):
        super().__init__()
        self.norm_1 = LayerNormalizer(n_embed)
        self.multi_attention = MultiHeadAttention(
            n_head, n_embed, attn_dropout, residual_dropout
        )

        self.norm_2 = LayerNormalizer(n_embed)
        self.ff = FeedForward(n_embed, dim_ff, residual_dropout)

    def forward(self, x, mask):
        att_input = self.norm_1(x)
        x = x + self.multi_attention(att_input, att_input, att_input, mask)
        x = x + self.ff(self.norm_2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed, attn_dropout, residual_dropout):
        super().__init__()
        if not n_embed % n_head == 0:
            raise DimensionError(
                "Number of heads must be a multiple of "
                "embedding dimension"
            )
        self.n_head = n_head
        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)

        # dropout
        self.dropout = nn.Dropout(residual_dropout)

        # attention
        self.attention = SelfAttention(attn_dropout)

        # projection layer
        self.projection_layer = nn.Linear(n_embed, n_embed)

    def forward(self, k, q, v, mask=None):
        B, T, C = k.shape
        # obtain key, query and value tensors
        k = self.key(k).view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)
        q = self.key(q).view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)
        v = self.key(v).view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)

        attention_scores = self.attention(k, q, v, mask)

        attention_scores = attention_scores.transpose(
            1, 2).contiguous().view(B, T, C)

        proj = self.projection_layer(attention_scores)
        return self.dropout(proj)


class SelfAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, k, q, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(k.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        attn_values = torch.matmul(attn_scores, v)

        return attn_values


class FeedForward(nn.Module):
    def __init__(self, n_embed, dim_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, dim_ff)
        self.linear_2 = nn.Linear(dim_ff, n_embed)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class LayerNormalizer(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.layer = nn.LayerNorm(n_embed)

    def forward(self, x):
        return self.layer(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, n_embed, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        weight = torch.zeros(max_len, n_embed)

        for position in range(max_len):
            for i in range(n_embed):
                # alternate sind and cosine values
                if i % 2 == 0:
                    value = math.sin(position / (10000 ** (2 * i / n_embed)))
                else:
                    value = math.cos(position / (10000 ** (2 * i / n_embed)))

                weight[position, i] = value

        self.weight = weight
        self.weight.requires_grad = False
        self.register_buffer('weight', weight)

    def forward(self, x):
        to_apply = self.weight.repeat(
            x.shape[0], 1, 1
        ).masked_fill(x == 0, 0)
        x += to_apply

        return self.dropout(x)
