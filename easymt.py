from utils.cli import get_arguments

from main_functions.split import split_file
from main_functions.train import train
from main_functions.preprocess import preprocess
from main_functions.vocab import build_vocab
from main_functions.translate import translate
from main_functions.evaluate import evaluate
from main_functions.convert_to_byte import convert_to_byte


def main():
    args = get_arguments()
    if args:
        subparser = args.subparser

        # chose functions
        if subparser == "split":
            split_file(args)

        elif subparser == "train":
            train(args)

        elif subparser == "preprocess":
            preprocess(args)

        elif subparser == "build-vocab":
            build_vocab(args)

        elif subparser == "translate":
            translate(args)

        elif subparser == "evaluate":
            evaluate(args)

        elif subparser == "convert-to-byte":
            convert_to_byte(args)


if __name__ == "__main__":
    main()
