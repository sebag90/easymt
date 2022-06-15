"""
split a TSV document containing 2 languages
into 2 documents containing each a single language
"""

from pathlib import Path


def main(args):
    print("Starting: Splitting document")

    filepath = Path(args.path)

    file1 = Path(f"{filepath}.l1")
    file2 = Path(f"{filepath}.l2")

    with open(filepath, "r", encoding="utf-8") as infile:
        l1file = open(file1, "w", encoding="utf-8")
        l2file = open(file2, "w", encoding="utf-8")

        for i, line in enumerate(infile):
            l1, l2 = line.strip().split("\t")

            l1file.write(f"{l1}\n")
            l2file.write(f"{l2}\n")

            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", flush=True)

        l1file.close()
        l2file.close()

    print("Complete: Splitting document")
