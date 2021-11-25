import configparser

from utils.lang import Vocab


def build_vocab(args):
    config = configparser.ConfigParser()
    config.read(args.path)
    src = config["DATASET"]["source"]
    tgt = config["DATASET"]["target"]
    min_freq = int(config["DATASET"]["min_freq"])
    max_len = int(config["MODEL"]["max_length"])
    n_sample = int(args.n_sample)

    voc1 = Vocab(src, min_freq)
    voc2 = Vocab(tgt, min_freq)

    l1_file = f"data/train.{src}"
    l2_file = f"data/train.{tgt}"

    with open(l1_file, "r", encoding="utf-8") as srcvoc, \
            open(l2_file, "r", encoding="utf-8") as tgtvoc:
        for i, (l1, l2) in enumerate(zip(srcvoc, tgtvoc)):

            l1 = l1.strip()
            l2 = l2.strip()

            # enforce max len
            if len(l1.split()) and len(l2.split()) <= max_len:
                voc1.add_sentence(l1)
                voc2.add_sentence(l2)

            print(f"Building vocabulary -- line: {i + 1}", end="\r")

            if n_sample != 0:
                if i > n_sample:
                    break

    voc1.save_voc()
    voc2.save_voc()
