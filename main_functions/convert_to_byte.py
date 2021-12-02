import configparser
import os
from pathlib import Path
import pickle

from utils.lang import Language
from utils.dataset import DataLoader


def convert_to_byte(args):
    # read data from configuration file
    config = configparser.ConfigParser()
    config.read(args.path)
    batches_per_file = args.n
    src = config["DATASET"]["source"]
    tgt = config["DATASET"]["target"]
    max_len = int(
        config["MODEL"]["max_length"]
    )
    batch_size = int(
        config["TRAINING"]["batch_size"]
    )

    # create language objects
    src_language = Language(src)
    tgt_language = Language(tgt)

    # read vocabulary from file
    src_language.read_vocabulary(
        Path(f"data/vocab.{src_language.name}")
    )
    tgt_language.read_vocabulary(
        Path(f"data/vocab.{tgt_language.name}")
    )

    # load entire dataset
    train_data = DataLoader.from_files(
        "train", src_language, tgt_language,
        max_len, batch_size
    )

    # find last file in data/batched
    os.makedirs(Path("data/batched"), exist_ok=True)
    existing = os.listdir(Path("data/batched"))

    if len(existing) == 0:
        start_from = 0
    else:
        start_from = max([int(i) for i in existing])

    batches = list()
    for i, batch in enumerate(train_data):
        batches.append(batch)

        if len(batches) == batches_per_file:
            # write batches to file
            to_write = Path(f"data/batched/{start_from}_{len(train_data)}")
            with open(to_write, "wb") as ofile:
                pickle.dump(batches, ofile)

            # increment name of file and empty batches list
            start_from += 1
            batches = list()

        # print progress
        print(f"Saving batch: {i}/{len(train_data)}", end="\r")

    # save last incomplete list of batches
    to_write = Path(f"data/batched/{start_from}_{len(train_data)}")
    with open(to_write, "wb") as ofile:
        pickle.dump(batches, ofile)

    print("Batching process complete!")
