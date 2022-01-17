import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.errors import DimensionError


class DecoderLayer(nn.Module):
    def __init__(
            self, d_model, n_head, dim_ff, attn_dropout, residual_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.masked_attn = MultiHeadAttention(
            n_head, d_model, attn_dropout, residual_dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            n_head, d_model, attn_dropout, residual_dropout
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_ff, residual_dropout)

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
            self, d_model, n_head, dim_ff, attn_dropout, residual_dropout):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.multi_attention = MultiHeadAttention(
            n_head, d_model, attn_dropout, residual_dropout
        )

        self.norm_2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_ff, residual_dropout)

    def forward(self, x, mask):
        att_input = self.norm_1(x)
        x = x + self.multi_attention(att_input, att_input, att_input, mask)
        x = x + self.ff(self.norm_2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, attn_dropout, residual_dropout):
        super().__init__()
        if not d_model % n_head == 0:
            raise DimensionError(
                "Number of heads must be a multiple of "
                "embedding dimension"
            )
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # dropout
        self.dropout = nn.Dropout(residual_dropout)

        # scaled dot product
        self.scaled_dot_product = ScaledDotProduct(attn_dropout)

        # projection layer
        self.projection_layer = nn.Linear(d_model, d_model)

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

        attention_scores = self.scaled_dot_product(q, k, v, mask)

        # concatenate head outputs
        attention_scores = attention_scores.transpose(
            1, 2).contiguous().view(B, T, C)

        proj = self.projection_layer(attention_scores)
        return self.dropout(proj)


class ScaledDotProduct(nn.Module):
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
    def __init__(self, d_model, dim_ff, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create sin/cos matrix
        den = torch.exp(
            -torch.arange(0, d_model, 2) * math.log(10000) / d_model
        )
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        pos_embedding.requires_grad = False
        self.register_buffer('weight', pos_embedding)

    def forward(self, x):
        to_apply = self.weight[:, :x.size(1)]
        x = x + to_apply
        return self.dropout(x)
