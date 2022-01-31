from functools import total_ordering
import math
import os
import pickle
from pathlib import Path
import random
import re


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
    def __init__(self, src, tgt, batch_size=32):
        if not len(src) == len(tgt):
            raise ValueError
        self.batch_size = batch_size
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

        self.shuffle()

    def shuffle(self):
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
    def from_files(cls, src_file, tgt_file, max_len, batch_size):
        """
        read src and tgt file and prepare dataset of encoded
        sentences in pairs (src, tgt)
        """
        src = list()
        tgt = list()
        src_file = Path(src_file)
        tgt_file = Path(tgt_file)

        # read data and create dataset
        with open(src_file, "r", encoding="utf-8") as inlang, \
                open(tgt_file, "r", encoding="utf-8") as outlang:
            for l1, l2 in zip(inlang, outlang):
                l1 = l1.strip().split()
                l2 = l2.strip().split()

                if len(l1) <= max_len and len(l2) <= max_len:
                    src.append(l1)
                    tgt.append(l2)

        data = cls(src, tgt, batch_size=batch_size)
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

    def shuffle(self):
        """
        batched data cannot be shuffled, placeholder
        """
        pass

    def __iter__(self):
        for entry in os.scandir(self.path):
            with open(Path(f"{self.path}/{entry.name}"), "rb") as infile:
                batches = pickle.load(infile)

            # batches is always a list of batches
            for batch in batches:
                yield batch
