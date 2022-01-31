"""
Train dataset will be converted to to bytes.
Each file contains N batches (already converted to indeces).
N can be modified. This is useful to reduce memory
requirements during training since files are loaded
sequentially.
"""

from pathlib import Path
import pickle

from utils.dataset import DataLoader
from utils.parameters import Parameters


def batch_dataset(args):
    # read data from configuration file
    config = Parameters.from_config(args.path)

    # load entire dataset
    train_data = DataLoader.from_files(
        config.data.src_train,
        config.data.tgt_train,
        config.model.max_length,
        config.training.batch_size
    )

    outputfile = Path(args.output)
    with open(outputfile, "w", encoding="utf-8") as ofile:
        for i, batch in enumerate(train_data):
            for src, tgt in zip(*batch):
                s_sen = " ".join(src)
                t_sen = " ".join(tgt)
                ofile.write(f"{s_sen}\t{t_sen}\n")

            # print progress
            print(f"Batching dataset: {i}/{len(train_data)}", end="\r")

    print(" " * 50, end="\r")
    print("Batching dataset: complete")
