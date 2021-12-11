"""
split a TSV document containing 2 languages
into 2 documents containing each a single language
"""

from pathlib import Path
import os


def split_file(args):
    path = Path(args.path)

    file1 = Path(f"{path}.l1")
    file2 = Path(f"{path}.l2")

    if os.path.exists(file1):
        os.remove(file1)

    if os.path.exists(file2):
        os.remove(file2)

    line_count = 0
    with open(path, "r", encoding="utf-8") as infile:
        l1file = open(file1, "a", encoding="utf-8")
        l2file = open(file2, "a", encoding="utf-8")

        for i, line in enumerate(infile):
            l1, l2 = line.strip().split("\t")

            l1file.write(f"{l1}\n")
            l2file.write(f"{l2}\n")

            if i % 5000 == 0:
                line_count += 5000
                print(
                    f"Splitting document: line {line_count}",
                    end="\r"
                )

        l1file.close()
        l2file.close()

    print(" " * 50)
    print("Splitting document: complete")
