from pathlib import Path

from utils.utils import split_filename


def slice_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main(args):
    print("Starting: Chunking")

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

            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", flush=True)

    print("Complete: Chunking")
