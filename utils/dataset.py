from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from functools import total_ordering
import math
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


class DataLoader(list):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.bins = defaultdict(list)

    @property
    def n_batches(self):
        return math.ceil(len(self) // self.batch_size)

    def add_pair(self, src, tgt):
        p = Pair(src, tgt)
        position = len(self)
        self.append(p)

        # add to bin with same length
        self.bins[p.len].append(position)

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

    def __iter__(self):
        for i in range(0, len(self.order), self.batch_size):
            batch_idx = self.order[i:i + self.batch_size]
            src = [self[idx].src.split() for idx in batch_idx]
            tgt = [self[idx].tgt.split() for idx in batch_idx]
            yield src, tgt

    def n_longest(self, n):
        yielded = 0
        for key in sorted(self.bins.keys(), reverse=True):
            for position in self.bins[key]:
                yield self[position]
                yielded += 1

                if yielded == n:
                    return

    @classmethod
    def from_files(cls, src_file, tgt_file, max_len, batch_size):
        """
        read src and tgt file and prepare dataset of encoded
        sentences in pairs (src, tgt)
        """
        data = cls(batch_size=batch_size)

        # language modeling
        if tgt_file is None:
            with Path(src_file).open(encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()

                    if 0 < len(line.split()) <= max_len:
                        data.add_pair(line, "")

        # language translation
        else:
            src_file = Path(src_file)
            tgt_file = Path(tgt_file)

            # read data and create dataset
            with src_file.open("r", encoding="utf-8") as inlang, \
                    tgt_file.open("r", encoding="utf-8") as outlang:
                for l1, l2 in zip(inlang, outlang):
                    l1 = l1.strip()
                    l2 = l2.strip()

                    if ((0 < len(l1.split()) <= max_len)
                            and (0 < len(l2.split()) <= max_len)):
                        data.add_pair(l1, l2)

        data.shuffle()
        return data


class BatchedData:
    def __init__(self, path, max_len, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.max_len = max_len

    def __repr__(self):
        return f"BatchedData({self.path})"

    def shuffle(self):
        """
        batched data cannot be shuffled, placeholder
        """
        pass

    def __iter__(self):
        with Path(self.path).open("r", encoding="utf-8") as infile:
            # read a batch from the dataset file
            batch = list(islice(infile, self.batch_size))
            while len(batch) != 0:
                src = list()
                tgt = list()
                for line in batch:
                    line = line.strip()
                    if len(line) > 0:
                        line = line.split("\t")

                        # language modeling data
                        if len(line) == 1:
                            src.append(line[0].split())
                            tgt.append("")

                        # language translation data
                        else:
                            s, t = line.split("\t")
                            src.append(s.split())
                            tgt.append(t.split())

                if len(src) > 0 and len(tgt) > 0:
                    yield src, tgt

                # get next batch
                batch = list(islice(infile, self.batch_size))
