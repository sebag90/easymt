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
from main_functions.clean import clean
from main_functions.number_replacer import replace_numbers


def main():
    args = get_arguments()
    if args:
        subparser = args.subparser

        functions = {
            "split": split_file,
            "clean": clean,
            "preprocess": preprocess,
            "split-dataset": split_dataset,
            "build-vocab": build_vocab,
            "convert-to-byte": convert_to_byte,
            "train": train,
            "translate": translate,
            "normalize": normalize,
            "evaluate": evaluate,
            "replace-numbers": replace_numbers
        }

        if subparser not in functions.keys():
            raise ValueError(
                "Invalid option"
            )

        fn = functions[subparser]
        fn(args)


if __name__ == "__main__":
    main()
