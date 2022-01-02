from pathlib import Path

from preprocessing_tools.dexmler import Dexmler
from preprocessing_tools.cleaner import Cleaner

from utils.utils import split_filename


def clean(args):
    # get arguments
    file_1 = Path(args.file1)
    file_2 = Path(args.file2)

    dexmler = Dexmler()
    cleaner = Cleaner(
        min_len=args.min_len,
        max_len=args.max_len,
        ratio=args.ratio
    )

    # create output files
    f1_path, f1_name, f1_suffix = split_filename(str(file_1))
    ofile1 = open(
        Path(f"{f1_path}/{f1_name}.clean.{f1_suffix}"), "w", encoding="utf-8"
    )

    f2_path, f2_name, f2_suffix = split_filename(str(file_2))
    ofile2 = open(
        Path(f"{f2_path}/{f2_name}.clean.{f2_suffix}"), "w", encoding="utf-8"
    )

    with open(file_1, "r", encoding="utf-8") as f1, \
            open(file_2, "r", encoding="utf-8") as f2:
        for i, (line_1, line_2) in enumerate(zip(f1, f2)):
            # clean files
            dexmled1, dexmled2 = dexmler(line_1.strip(), line_2.strip())
            cleaned1, cleaned2 = cleaner(dexmled1, dexmled2)

            if cleaned1 != "" and cleaned2 != "":
                # write to files
                ofile1.write(f"{cleaned1}\n")
                ofile2.write(f"{cleaned2}\n")

            print(f"Cleaning: line {i:,}", end="\r")

    # close output files
    ofile1.close()
    ofile2.close()

    print(" "*50, end="\r")
    print("Cleaning: complete")
