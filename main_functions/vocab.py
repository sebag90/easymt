from utils.lang import Vocab
from utils.parameters import Parameters


def build_vocab(args):
    config = Parameters.from_config(args.path)
    n_sample = int(args.n_sample)

    voc1 = Vocab(config.dataset.source, config.dataset.min_freq)
    voc2 = Vocab(config.dataset.target, config.dataset.min_freq)

    l1_file = f"data/train.{config.dataset.source}"
    l2_file = f"data/train.{config.dataset.target}"

    with open(l1_file, "r", encoding="utf-8") as srcvoc, \
            open(l2_file, "r", encoding="utf-8") as tgtvoc:
        for i, (l1, l2) in enumerate(zip(srcvoc, tgtvoc)):

            l1 = l1.strip()
            l2 = l2.strip()

            # enforce max len
            if len(l1.split()) and len(l2.split()) <= config.model.max_length:
                voc1.add_sentence(l1)
                voc2.add_sentence(l2)

            print(f"Building vocabulary: line: {i + 1}", end="\r")

            if n_sample != 0:
                if i > n_sample:
                    break

    voc1.save_voc()
    voc2.save_voc()

    print(" "*100, end="\r")
    print("Building vocabulary: complete")
