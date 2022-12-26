import argparse


def easymt_arguments():
    parser = argparse.ArgumentParser(
        description="EasyMT: Neural Machine Translation",
        prog="easymt"
    )
    subparsers = parser.add_subparsers(dest="subparser")

    # build vocab
    vocab = subparsers.add_parser(
        "vocab",
        help="build a vocabulary from files"
    )
    vocab.add_argument(
        "--progress", "-p", action="store_true",
        help="print progress status"
    )
    vocab.add_argument(
        "--n-sample", metavar="N", action="store",
        help=(
            "number of lines used to build vocabulary"
            " (0 = full corpus - default: %(default)s)"
        ),
        type=int,
        default=0
    )
    vocab.add_argument(
        "--min-freq", metavar="N", type=int,
        help=(
            "minimum frequency for a token to be included"
            " (default: %(default)s)"
        ),
        default=2
    )

    # batch data set
    batch = subparsers.add_parser(
        "batch",
        help="convert train files to byte files"
    )
    batch.add_argument(
        "path", action="store",
        help="path to the configuration file"
    )
    batch.add_argument(
        "--output-max",  action="store",
        help="output file"
    )
    batch.add_argument(
        "--max", metavar="N", action="store",
        help=(
            "saves a sample file containing the N longest "
            "sequences in the dataset"
        ),
        default=0, type=int
    )

    # train
    train = subparsers.add_parser(
        "train", help="train a new model"
    )
    train.add_argument(
        "path", metavar="PATH", action="store",
        help="path to the configuration file"
    )
    train.add_argument(
        "--resume", action="store", metavar="MODEL",
        help="path to model to resume training"
    )
    train.add_argument(
        "--batched", action="store", metavar="BATCHED-FILE",
        help=(
            "path to the directory containing "
            "the batched files")
    )
    train.add_argument(
        "--mixed", action="store_true",
        help="Train the model with mixed precision"
    )

    # translate
    translate = subparsers.add_parser(
        "translate", help="translate with a trained model"
    )
    translate.add_argument(
        "--model", action="store",
        help="path to model",
        required=True
    )
    translate.add_argument(
        "--method", metavar="METHOD", action="store",
        choices=["beam", "topk"],
        help="Decoding algorithm (default: %(default)s)"
    )
    translate.add_argument(
        "--beam", metavar="N", action="store",
        default=5,
        help="size of search beam (default: %(default)s)"
    )
    translate.add_argument(
        "--k", metavar="N", action="store",
        default=10, type=int,
        help="K value for top-K algorithm (default: %(default)s)"
    )
    translate.add_argument(
        "--steps", metavar="N", action="store",
        default=150, type=int,
        help="Number of steps while decoding (default: %(default)s)"
    )
    translate.add_argument(
        "--temperature", metavar="N", action="store",
        default=1,
        help="Temperature for the Top K algorithm (default: %(default)s)"
    )
    translate.add_argument(
        "--verbose", action="store_true",
        help="print all candidates for each line"
    )
    translate.add_argument(
        "--alpha", action="store",
        help="weighing factor for repetition in hypothesis",
        default=0, type=float
    )

    args = parser.parse_args()
    if args.subparser not in subparsers.choices.keys():
        parser.print_help()
        return
    return args
