from functools import total_ordering
import math
import os
import pickle
from pathlib import Path
import random
import re

import torch


@total_ordering
class Pair:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        self.len = len(src)

    def __lt__(self, other):
        return self.len < other.len

    def __repr__(self):
        return str((self.src, self.tgt))

    def __len__(self):
        return self.len


class DataLoader:
    def __init__(self, src, tgt, shuffle=False, batch_size=32):
        if not len(src) == len(tgt):
            raise ValueError
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = list()
        self.bins = dict()

        # create pair objects
        for i, (src_sen, tgt_sen) in enumerate(zip(src, tgt)):
            p = Pair(src_sen, tgt_sen)
            self.data.append(p)

            # add to bin with same length
            if p.len not in self.bins:
                self.bins[p.len] = list()
            self.bins[p.len].append(i)

        self.create_order()

    def create_order(self):
        """
        Order is based on length of input sentence.
        First the keys of self.bins are shuffled and then
        each bin is shuffled and added to self.order
        """
        self.order = list()
        # shuffle bins keys (lenght of sentences)
        keys = list(self.bins.keys())
        random.shuffle(keys)
        # shuffle indeces at each key and add to self.order
        for key in keys:
            to_add = self.bins[key]
            shuffled = random.sample(to_add, len(to_add))
            self.order += shuffled

    def __len__(self):
        return math.ceil(len(self.data) // self.batch_size)

    def __iter__(self):
        for i in range(0, len(self.order), self.batch_size):
            batch_idx = self.order[i:i + self.batch_size]
            src = [self.data[i].src for i in batch_idx]
            tgt = [self.data[i].tgt for i in batch_idx]
            yield src, tgt

    @classmethod
    def from_files(cls, name, src_language, tgt_language, max_len, batch_size):
        """
        read src and tgt file and prepare dataset of encoded
        sentences in pairs (src, tgt)
        """
        src = list()
        tgt = list()
        src_file = Path(f"data/{name}.{src_language.name}")
        tgt_file = Path(f"data/{name}.{tgt_language.name}")

        # read data and create dataset
        with open(src_file, "r", encoding="utf-8") as inlang, \
                open(tgt_file, "r", encoding="utf-8") as outlang:
            for l1, l2 in zip(inlang, outlang):
                l1 = l1.strip()
                l2 = l2.strip()

                if len(l1.split()) <= max_len and len(l2.split()) <= max_len:
                    # convert sentence to vector
                    l1_coded = src_language.indexes_from_sentence(l1)
                    l2_coded = tgt_language.indexes_from_sentence(l2)

                    # save coded vectors to create dataset
                    src.append(l1_coded)
                    tgt.append(l2_coded)

        data = cls(src, tgt, shuffle=True, batch_size=batch_size)
        return data


class BatchedData:
    def __init__(self, path):
        self.path = path
        self.len = 0
        # obtain total number of batches from last file
        num = re.compile(r"_([0-9]+)")
        for entry in os.scandir(path):
            length = re.search(num, entry.name)
            if length is not None:
                n = int(length.group(1))
                if n >= self.len:
                    self.len = n

    def __len__(self):
        return self.len

    def create_order(self):
        """
        do nothing, placeholder
        """
        pass

    def __iter__(self):
        for entry in os.scandir(self.path):
            with open(Path(f"data/batched/{entry.name}"), "rb") as infile:
                batches = pickle.load(infile)

            # batches is always a list of batches
            for batch in batches:
                yield batch


class RNNDataTransformer:
    def __call__(self, src, tgt, max_len, sos_index_tgt_lang):
        # prepare source data
        src_len = torch.tensor([len(indexes) for indexes in src])
        src_pad = torch.nn.utils.rnn.pad_sequence(src)

        # prepare target data
        max_tgt_len = max([len(indexes) for indexes in tgt])
        tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt)

        # prepare mask
        mask = tgt_pad != 0

        return src_pad, src_len, tgt_pad, mask, max_tgt_len


class TransformerDataconverter:
    def __call__(self, src, tgt, max_len, sos_index_tgt_lang):
        # create target variables by removing <eos> token
        decoder_input = list()
        for sentence in tgt:
            decoder_input.append(sentence[:-1])

        decoder_input = torch.nn.utils.rnn.pad_sequence(
            decoder_input, batch_first=True
        )
        src = torch.nn.utils.rnn.pad_sequence(
            src, batch_first=True
        )
        target = torch.nn.utils.rnn.pad_sequence(
            tgt, batch_first=True
        )

        # add <sos> padding to decoder input
        sos_padder = torch.nn.ConstantPad2d((1, 0, 0, 0), sos_index_tgt_lang)
        decoder_input = sos_padder(decoder_input)

        # create masks
        e_mask = (src != 0).unsqueeze(1)
        d_mask = (decoder_input != 0).unsqueeze(1)
        subseq_mask = torch.ones(
            (1, decoder_input.size(1), decoder_input.size(1)),
            dtype=torch.bool
        )
        subseq_mask = torch.tril(subseq_mask)
        d_mask = torch.logical_and(d_mask, subseq_mask)

        return src, decoder_input, target, e_mask, subseq_mask
