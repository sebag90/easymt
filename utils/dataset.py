import itertools
from functools import total_ordering
from pathlib import Path
import random

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

    def zeroPadding(self, l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.order), self.batch_size):
            batch_idx = self.order[i:i + self.batch_size]

            # prepare source data
            src = [self.data[i].src for i in batch_idx]
            src_len = torch.tensor([len(indexes) for indexes in src])
            padlist = self.zeroPadding(src)
            src_pad = torch.tensor(padlist)

            # prepare target data
            tgt = [self.data[i].tgt for i in batch_idx]
            max_tgt_len = max([len(indexes) for indexes in tgt])
            padlist = self.zeroPadding(tgt)
            tgt_pad = torch.tensor(padlist)
            mask = tgt_pad.clone()
            mask[mask != 0] = 1
            mask = mask.bool()

            yield src_pad, src_len, tgt_pad, mask, max_tgt_len

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
