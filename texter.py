from utils.cli import texter_arguments

from main_functions.split import split_file
from main_functions.preprocess import preprocess
from main_functions.chunk import chunk
from main_functions.split_dataset import split_dataset
from main_functions.normalize import normalize
from main_functions.clean import clean
from main_functions.number_replacer import replace_numbers
from main_functions.evaluate import evaluate


def main():
    args = texter_arguments()
    if args:
        subparser = args.subparser

        functions = {
            "split": split_file,
            "clean": clean,
            "preprocess": preprocess,
            "split-dataset": split_dataset,
            "normalize": normalize,
            "replace-numbers": replace_numbers,
            "evaluate": evaluate,
            "chunk": chunk
        }

        if subparser not in functions.keys():
            raise ValueError(
                "Invalid option"
            )

        fn = functions[subparser]
        fn(args)


if __name__ == "__main__":
    main()
