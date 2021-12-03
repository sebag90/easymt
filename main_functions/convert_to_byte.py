import configparser
import os
from pathlib import Path
import pickle
import re

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

    # create output folder to save batched files
    output_path = Path("data/batched")
    os.makedirs(output_path, exist_ok=True)

    # find last file in data/batched
    # and number batches in already saved files
    file_num = re.compile(r"[0-9]+")
    batch_num = re.compile(r"_([0-9]+)")
    start_from = 0
    total_batches = 0
    for entry in os.scandir(output_path):
        # search for numbers in file name
        found_file_n = re.match(file_num, entry.name)
        found_batch_n = re.search(batch_num, entry.name)

        # update highest numbers
        if found_file_n is not None and found_batch_n is not None:
            found_f = int(found_file_n.group())
            found_b = int(found_batch_n.group(1))

            # update file number
            if found_f >= start_from:
                start_from = found_f + 1

            # update number of batches
            if found_b > total_batches:
                total_batches = found_b

    # start saving batches
    batches = list()
    total_len = len(train_data) + total_batches
    for i, batch in enumerate(train_data):
        batches.append(batch)

        if len(batches) == batches_per_file:
            # write batches to file
            to_write = Path(f"{output_path}/{start_from}_{total_len}")
            with open(to_write, "wb") as ofile:
                pickle.dump(batches, ofile)

            # increment name of file and empty batches list
            start_from += 1
            batches = list()

        # print progress
        print(f"Saving batch: {i}/{len(train_data)}", end="\r")

    # save last incomplete list of batches
    to_write = Path(f"data/batched/{start_from}_{total_len}")
    with open(to_write, "wb") as ofile:
        pickle.dump(batches, ofile)

    print("Batching process complete!")
