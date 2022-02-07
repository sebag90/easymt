from pathlib import Path

from utils.utils import split_filename


def slice_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def chunk(args):
    input_file = Path(args.file)
    max_len = args.n

    path, name, suffix = split_filename(str(input_file))
    output_file = Path(f"{path}/{name}.chunked.{suffix}")

    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as ofile:
        for i, line in enumerate(infile):
            chunks = slice_list(line.strip().split(), max_len)
            for sen in chunks:
                to_write = " ".join(sen)
                ofile.write(f"{to_write}\n")

            print(f"Chunking: line {i:,}", end="\r")

    print(" "*50, end="\r")
    print("Chunking: complete")
