import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, src_lang, tgt_lang, word_vec_size, shared):
        super().__init__()
        src_vocab = len(src_lang)
        tgt_vocab = len(tgt_lang)

        if shared is False:
            src_pad_id = src_lang.word2index["<pad>"]
            tgt_pad_id = tgt_lang.word2index["<pad>"]

            self.src = nn.Embedding(
                src_vocab,
                word_vec_size,
                padding_idx=src_pad_id
            )
            self.tgt = nn.Embedding(
                tgt_vocab,
                word_vec_size,
                padding_idx=tgt_pad_id
            )

        else:
            # make sure 2 embeddings share same pad index
            src_pad_id = src_lang.word2index["<pad>"]
            tgt_pad_id = tgt_lang.word2index["<pad>"]
            assert src_pad_id == tgt_pad_id

            # make sure 2 embeddings have same vocab length
            assert src_vocab == tgt_vocab

            self.src = self.tgt = nn.Embedding(
                tgt_vocab,
                word_vec_size,
                padding_idx=src_pad_id
            )
