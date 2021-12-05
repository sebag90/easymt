from pathlib import Path

from preprocessing_tools.dexmler import Dexmler
from preprocessing_tools.cleaner import Cleaner

from utils.utils import count_lines, name_suffix_from_file
from utils.errors import FileError


def clean(args):
    # get arguments
    file_1 = Path(args.file1)
    file_2 = Path(args.file2)
    max_len = args.len

    dexmler = Dexmler()
    cleaner = Cleaner(1, max_len)

    # make sure files have same number of lines
    len_f1 = count_lines(file_1)
    len_f2 = count_lines(file_2)

    if len_f1 != len_f2:
        raise FileError(
            "Documents have differents lengths"
        )

    # create output files
    f1_name, f1_suffix = name_suffix_from_file(str(file_1))
    ofile1 = open(
        Path(f"{f1_name}_clean.{f1_suffix}"), "w", encoding="utf-8"
    )

    f2_name, f2_suffix = name_suffix_from_file(str(file_2))
    ofile2 = open(
        Path(f"{f2_name}_clean.{f2_suffix}"), "w", encoding="utf-8"
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

            print(f"Cleaning: line {i}", end="\r")

    # close output files
    ofile1.close()
    ofile2.close()

    print(" "*100, end="\r")
    print("Cleaning: complete")
