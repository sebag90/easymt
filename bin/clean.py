from pathlib import Path

from preprocessing_tools.dexmler import Dexmler
from preprocessing_tools.cleaner import Cleaner

from utils.utils import split_filename
from utils.errors import InvalidArgument


def main(args):
    print("Starting: Cleaning")

    # get arguments
    if not 0 < len(args.file) < 3:
        raise InvalidArgument(
            "You can only pass either one or two files"
        )

    files = [Path(name) for name in args.file]

    dexmler = Dexmler()
    cleaner = Cleaner(
        min_len=args.min_len,
        max_len=args.max_len,
        ratio=args.ratio
    )

    # create output files
    output_files = list()
    for filepath in files:
        path, name, suffix = split_filename(str(filepath))
        output_files.append(
            open(Path(f"{path}/{name}.clean.{suffix}"), "w", encoding="utf-8")
        )

    # open read files
    input_files = list()
    for filepath in files:
        input_files.append(open(filepath))

    for i, lines in enumerate(zip(*input_files)):
        # clean files
        dexmled = dexmler(*lines)
        cleaned = cleaner(*dexmled)

        if all((i != "" for i in cleaned)):
            for line, ofile in zip(cleaned, output_files):
                ofile.write(f"{line}\n")

        if (i+1) % 10000 == 0:
            print(f"Processed lines: {i + 1:,}", flush=True)

    # close all files
    for open_file in output_files + input_files:
        open_file.close()

    print("Complete: Cleaning")
