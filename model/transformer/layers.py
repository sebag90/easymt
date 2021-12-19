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
        self.d_k = n_embed // n_head
        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)

        # dropout
        self.dropout = nn.Dropout(residual_dropout)

        # attention
        self.attention = SelfAttention(attn_dropout)

        # projection layer
        self.projection_layer = nn.Linear(n_embed, n_embed)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape
        # obtain key, query and value tensors
        k = self.key(k).view(
            B, -1, self.n_head, self.d_k
        ).transpose(1, 2)
        q = self.key(q).view(
            B, -1, self.n_head, self.d_k
        ).transpose(1, 2)
        v = self.key(v).view(
            B, -1, self.n_head, self.d_k
        ).transpose(1, 2)

        attention_scores = self.attention(q, k, v, mask)

        attention_scores = attention_scores.transpose(
            1, 2).contiguous().view(B, -1, self.n_head * self.d_k)

        proj = self.projection_layer(attention_scores)
        return self.dropout(proj)


class SelfAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        values = torch.matmul(scores, v)
        return values


class FeedForward(nn.Module):
    def __init__(self, n_embed, dim_ff, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embed, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


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

        # create sin/cos matrix
        den = torch.exp(
            -torch.arange(0, n_embed, 2) * math.log(10000) / n_embed
        )
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, n_embed))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        pos_embedding.requires_grad = False
        self.register_buffer('weight', pos_embedding)

    def forward(self, x):
        to_apply = self.weight[:, :x.size(1)]
        x = x + to_apply
        return self.dropout(x)
