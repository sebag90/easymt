"""
Train dataset will be converted to to bytes.
Each file contains N batches (already converted to indeces).
N can be modified. This is useful to reduce memory
requirements during training since files are loaded
sequentially.
"""

from pathlib import Path
import random
import sys

from utils.dataset import DataLoader, EmptyFile


def main(args):
    if args.output is not None:
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        output_file = sys.stdout

    data = DataLoader(batch_size=args.batch_size)

    src_file = Path(args.src).open(encoding="utf-8")
    if args.tgt is None:
        tgt_file = EmptyFile()
    else:
        tgt_file = Path(args.tgt).open(encoding="utf-8")

    for i, (l1, l2) in enumerate(zip(src_file, tgt_file)):
        l1 = l1.strip()
        l2 = l2.strip()

        if args.tgt is None:
            can_add = args.min_len < len(l1.split()) <= args.max_len
        else:
            can_add = ((args.min_len < len(l1.split()) <= args.max_len) and
                       (args.min_len < len(l2.split()) <= args.max_len))

        if can_add:
            to_write = data.add_pair(l1, l2, dynamic=True)
            if to_write is not None:
                random.shuffle(to_write)
                for pair in to_write:
                    print(f"{pair.src}\t{pair.tgt}", file=output_file)

        if i % 1000000 == 0:
            print(i, end="", file=sys.stderr, flush=True)

        elif i % 100000 == 0:
            print(".", end="", file=sys.stderr, flush=True)

    # write remaining data in dataloader
    data.shuffle()
    for batch in data:
        for src, tgt in zip(*batch):
            s_sen = " ".join(src)
            t_sen = " ".join(tgt)
            print(f"{s_sen}\t{t_sen}", file=output_file)

    src_file.close()
    tgt_file.close()
    output_file.close()
    print("\nComplete: Batching dataset", file=sys.stderr)
