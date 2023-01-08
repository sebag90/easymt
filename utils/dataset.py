from collections import defaultdict
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
import random


@dataclass
@total_ordering
class Pair:
    src: str
    tgt: str

    def __post_init__(self):
        self.len = len(self.src.split())

    def __lt__(self, other):
        return self.len < other.len

    def __len__(self):
        return self.len


class EmptyFile:
    def __iter__(self):
        while True:
            yield ""

    def close(self):
        pass

    def open(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DataLoader(dict):
    def __init__(self, src_file, tgt_file, max_len, batch_size=32):
        self.batch_size = batch_size
        self.bins = defaultdict(list)
        self.buffer = list()
        self.i = 0
        self.max_len = max_len
        self.tgt_file = Path(tgt_file) if tgt_file is not None else EmptyFile()
        self.src_file = Path(src_file)

    def add_pair(self, src, tgt):
        p = Pair(src, tgt)
        position = self.i
        self[position] = p
        self.i += 1

        # add to bin with same length
        self.bins[p.len].append(position)

        if len(self.bins[p.len]) == self.batch_size:
            idxs = self.bins.pop(p.len)
            return [self.pop(i) for i in idxs]

        return None

    def shuffle(self):
        """
        Order is based on length of input sentence.
        First the keys of self.bins are shuffled and then
        each bin is shuffled and added to self.order.
        This ensures that sentences with the same length are
        batched together
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

    def empty_buffer(self):
        random.shuffle(self.buffer)
        buffer = self.buffer.copy()
        self.buffer = list()
        for batch in buffer:
            src = [datapoint.src.split() for datapoint in batch]
            tgt = [datapoint.tgt.split() for datapoint in batch]
            yield src, tgt

    def reset(self):
        self.i = 0
        self.clear()
        self.bins.clear()
        self.buffer = list()
        self.order = list()

    def __iter__(self):
        with self.src_file.open(encoding="utf-8") as srcfile, \
                self.tgt_file.open(encoding="utf-8") as tgtfile:
            for l1, l2 in zip(srcfile, tgtfile):
                # check if pair can be added to dataloader
                if isinstance(self.tgt_file, EmptyFile):
                    can_add = 0 < len(l1.split()) <= self.max_len
                else:
                    can_add = ((0 < len(l1.split()) <= self.max_len) and
                               (0 < len(l2.split()) <= self.max_len))

                # add pair
                if can_add:
                    batch = self.add_pair(l1, l2)
                    if batch is not None:
                        self.buffer.append(batch)

                # empty buffer if we have at least 10 batches
                if len(self.buffer) == 10:
                    yield from self.empty_buffer()

        # empty buffer before yielding unpaired batches
        yield from self.empty_buffer()

        # yield remaining sentences
        self.shuffle()
        for i in range(0, len(self.order), self.batch_size):
            batch_idx = self.order[i:i + self.batch_size]
            src = [self[idx].src.split() for idx in batch_idx]
            tgt = [self[idx].tgt.split() for idx in batch_idx]
            yield src, tgt

        self.reset()
