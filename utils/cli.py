import argparse


def easymt_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    # build vocab
    vocab = subparsers.add_parser(
        "build-vocab",
        help="build a vocabulary from files"
    )
    vocab.add_argument(
        "file", metavar="FILE(S)",
        nargs="+",
        action="store",
        help="path to file(s) to process"
    )
    vocab.add_argument(
        "--n-sample", metavar="N", action="store",
        help=(
            "number of lines used to build vocabulary"
            " (0 = full corpus - default: %(default)s)"
        ),
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
        "batch-dataset",
        help="convert train files to byte files"
    )
    batch.add_argument(
        "path", action="store",
        help="path to the configuration file"
    )
    batch.add_argument(
        "--output",  action="store",
        help="output file", required=True
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


def texter_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    # split TSV files
    split = subparsers.add_parser(
        "split-file", help="preprocess a TSV file"
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
        "file", metavar="FILE(S)",
        nargs="+",
        action="store",
        help="path to file(s) to be cleaned"
    )
    clean.add_argument(
        "--max-len", action="store",
        help="maximum sentence length (default: %(default)s)",
        default=70, type=int
    )
    clean.add_argument(
        "--min-len", action="store",
        help="minimum sentence length (default: %(default)s)",
        default=1, type=int
    )
    clean.add_argument(
        "--ratio", action="store",
        help="maximum ratio between length of sources (default: %(default)s)",
        default=9, type=int
    )

    chunker = subparsers.add_parser(
        "chunk", help="divide long sentences into chunks of n words"
    )
    chunker.add_argument(
        "file", metavar="FILE", action="store",
        help="path to the text file"
    )
    chunker.add_argument(
        "--max-len", metavar="N", action="store",
        type=int,
        help="maximum length of each sentence (default: %(default)s)",
        required=True,
        default=256
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
        metavar="BPE-Splits",
        help="number of BPE splittings",
        type=int
    )
    preprocess.add_argument(
        "--replace-nums", action="store_true",
        help="convert all numbers to <num>"
    )
    preprocess.add_argument(
        "--SP", action="store",
        metavar="V-Size",
        help=(
            "target vocabulary to be generated with sentencepiece "
            "if a model already exists in the same directory as the "
            "file, that model will be used instead of training a new one"
        ),
        type=int
    )
    preprocess.add_argument(
        "--max-lines", metavar="N", action="store",
        default=0, type=int,
        help="maximum number of lines used to train preprocessing models"
    )

    # split dataset
    split_dataset = subparsers.add_parser(
        "split-dataset",
        help="split a file in train, eval and test files"
    )
    split_dataset.add_argument(
        "file", metavar="FILE(S)",
        nargs="+",
        action="store",
        help="path to file(s) to be cleaned"
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
    normalize.add_argument(
        "--SP", action="store",
        metavar="V-Size",
        help="target vocabulary of the sentencepiece model"
    )
    normalize.add_argument(
        "--upper", "-u", action="store_true",
        help="uppercase the first char in the sentence"
    )

    # replace numbers
    replace_nums = subparsers.add_parser(
        "replace-numbers",
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
    evaluate.add_argument(
        "-lc", action="store_true",
        help="compute BLEU on lowercased text"
    )

    args = parser.parse_args()
    if args.subparser not in subparsers.choices.keys():
        parser.print_help()
        return
    return args
