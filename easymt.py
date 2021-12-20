from utils.cli import easymt_arguments

from main_functions.train import train
from main_functions.vocab import build_vocab
from main_functions.translate import translate
from main_functions.convert_to_byte import convert_to_byte


def main():
    args = easymt_arguments()
    if args:
        subparser = args.subparser

        functions = {
            "build-vocab": build_vocab,
            "convert-to-byte": convert_to_byte,
            "train": train,
            "translate": translate
        }

        if subparser not in functions.keys():
            raise ValueError(
                "Invalid option"
            )

        fn = functions[subparser]
        fn(args)


if __name__ == "__main__":
    main()
