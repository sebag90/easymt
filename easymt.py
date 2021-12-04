from utils.cli import get_arguments

from main_functions.split import split_file
from main_functions.train import train
from main_functions.preprocess import preprocess
from main_functions.vocab import build_vocab
from main_functions.translate import translate
from main_functions.evaluate import evaluate
from main_functions.convert_to_byte import convert_to_byte
from main_functions.split_dataset import split_dataset
from main_functions.normalize import normalize


def main():
    args = get_arguments()
    if args:
        subparser = args.subparser

        functions = {
            "split": split_file,
            "train": train,
            "preprocess": preprocess,
            "build-vocab": build_vocab,
            "translate": translate,
            "evaluate": evaluate,
            "convert-to-byte": convert_to_byte,
            "split-dataset": split_dataset,
            "normalize": normalize
        }

        if subparser not in functions.keys():
            raise ValueError(
                "Invalid option"
            )

        fn = functions[subparser]
        fn(args)


if __name__ == "__main__":
    main()
