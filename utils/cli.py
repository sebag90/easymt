import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    # split TSV files
    split = subparsers.add_parser(
        "split", help="preprocess a TSV file"
    )

    split.add_argument(
        "path", metavar="PATH", action="store",
        help=(
            "path to the TSV file"
        )
    )

    # clean 2 files
    clean = subparsers.add_parser(
        "clean", help="clean 2 files"
    )
    clean.add_argument(
        "file1", metavar="FILE1",
        action="store",
        help="file 1"
    )
    clean.add_argument(
        "file2", metavar="FILE2",
        action="store",
        help="file 2"
    )
    clean.add_argument(
        "--len", action="store",
        help="maximum file length (default: %(default)s)",
        required=True, default=50,
        type=int
    )

    # preprocess
    preprocess = subparsers.add_parser(
        "preprocess", help="preprocess a corpus"
    )
    preprocess.add_argument(
        "file", metavar="FILE", action="store",
        help="path to the text file"
    )
    pre_required = preprocess.add_argument_group(
        "required named arguments"
    )
    pre_required.add_argument(
        "--language", metavar="LANG",
        action="store",
        help="language of the file",
        required=True
    )
    preprocess.add_argument(
        "--bpe", action="store",
        help="number of BPE splittings",
        type=int
    )
    preprocess.add_argument(
        "--remove-nums", action="store_true",
        help="convert all numbers to <num>"
    )

    # split dataset
    split_dataset = subparsers.add_parser(
        "split-dataset",
        help="split a file in train, eval and test files"
    )
    split_dataset.add_argument(
        "l1", metavar="CORPUS L1",
        action="store",
        help="L1 corpus"
    )
    split_dataset.add_argument(
        "l2", metavar="CORPUS L2",
        action="store",
        help="L2 corpus"
    )
    split_required = split_dataset.add_argument_group(
        "required named arguments"
    )
    split_required.add_argument(
        "--train", metavar="N_train",
        action="store",
        help="number of lines for train data",
        required=True
    )
    split_required.add_argument(
        "--eval", metavar="N_eval",
        action="store",
        help="number of lines for evaluation data",
        required=True
    )
    split_required.add_argument(
        "--test", metavar="N_test",
        action="store",
        help="number of lines for test data",
        required=True
    )

    # build vocab
    vocab = subparsers.add_parser(
        "build-vocab",
        help="build a vocabulary from files"
    )
    vocab.add_argument(
        "file1", metavar="PATH", action="store",
        help="path to file 1"
    )
    vocab.add_argument(
        "file2", metavar="PATH", action="store",
        help="path to file 2"
    )
    vocab.add_argument(
        "--n_sample", metavar="N", action="store",
        help=(
            "number of lines used to build vocabulary"
            " (0 = full corpus - default: %(default)s)"
        ),
        default=0
    )
    vocab.add_argument(
        "--min_freq", metavar="N", type=int,
        help=(
            "minimum frequency for a token to be included"
            " (default: %(default)s)"
        ),
        default=2
    )

    # convert to byte
    byte = subparsers.add_parser(
        "convert-to-byte",
        help="convert train files to byte files"
    )
    byte.add_argument(
        "path", metavar="PATH", action="store",
        help="path to the configuration file"
    )
    byte.add_argument(
        "--n", metavar="N", action="store",
        default=100,
        help=(
            "number of batches for each file"
            " (Default: %(default)s)"
        )
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
        "--resume", action="store",
        help="path to model to resume training"
    )
    train.add_argument(
        "--batched", action="store_true",
        help="train files are already batched"
    )

    # translate
    translate = subparsers.add_parser(
        "translate", help="translate a file"
    )

    translate.add_argument(
        "file", metavar="FILE", action="store",
        help="path to file to translate"
    )
    translate.add_argument(
        "model", metavar="MODEL", action="store",
        help="path to model"
    )
    translate.add_argument(
        "--beam", metavar="N", action="store",
        default=5,
        help="size of search beam (default: %(default)s)"
    )
    translate.add_argument(
        "--verbose", action="store_true",
        help="print all candidates for each line"
    )

    # normalize
    normalize = subparsers.add_parser(
        "normalize",
        help="undo preprocessing to normalize text"
    )
    normalize.add_argument(
        "file", metavar="FILE",
        help="file to process"
    )
    normalize.add_argument(
        "--subword", action="store_true",
        help="subword splitting was applied"
    )

    # evaluate
    evaluate = subparsers.add_parser(
        "evaluate", help="compute BLEU score"
    )

    evaluate.add_argument(
        "reference", action="store",
        help="path to reference translation"
    )

    evaluate.add_argument(
        "translation", action="store",
        help="path to translated document"
    )

    # replace numbers
    replace_nums = subparsers.add_parser(
        "replace-nums",
        help="replace <num> tokens in tokenized file"
    )

    replace_nums.add_argument(
        "reference", action="store",
        help="path to reference translation"
    )

    replace_nums.add_argument(
        "translation", action="store",
        help="path to translated document"
    )

    args = parser.parse_args()
    if args.subparser not in subparsers.choices.keys():
        parser.print_help()
        return
    return args
