"""
Train dataset will be converted to to bytes.
Each file contains N batches (already converted to indeces).
N can be modified. This is useful to reduce memory
requirements during training since files are loaded
sequentially.
"""

from pathlib import Path
import sys

from utils.dataset import DataLoader
from utils.parameters import Parameters


def main(args):
    print("Starting: Batching dataset", file=sys.stderr)

    # read data from configuration file
    config = Parameters.from_config(args.path)

    if args.max != 0 and args.output_max is None:
        print(
            "you need to specify the outputfile "
            "of the file with the longest batches "
            "with the option --output-max",
            file=sys.stderr
        )
        return

    # load entire dataset
    train_data = DataLoader.from_files(
        config.data.src_train,
        config.data.tgt_train,
        config.model.max_length,
        config.training.batch_size
    )

    lines = 0
    for batch in train_data:
        for src, tgt in zip(*batch):
            s_sen = " ".join(src)
            t_sen = " ".join(tgt)
            print(f"{s_sen}\t{t_sen}", file=sys.stdout)
            lines += 1

            if lines % 1000000 == 0:
                print(lines, end="", file=sys.stderr, flush=True)

            elif lines % 100000 == 0:
                print(".", end="", file=sys.stderr, flush=True)

    if args.max != 0:
        max_file = Path(args.output_max)
        with max_file.open("w", encoding="utf-8") as ofile:
            for pair in train_data.n_longest(args.max):
                ofile.write(f"{pair.src}\t{pair.tgt}\n")

    print("\nComplete: Batching dataset", file=sys.stderr)