import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, word_vec_size, shared):
        super().__init__()
        if shared is False:
            self.src = nn.Embedding(
                src_vocab,
                word_vec_size,
                padding_idx=0
            )
            self.tgt = nn.Embedding(
                tgt_vocab,
                word_vec_size,
                padding_idx=0
            )

        else:
            self.src = self.tgt = nn.Embedding(
                tgt_vocab,
                word_vec_size,
                padding_idx=0
            )
