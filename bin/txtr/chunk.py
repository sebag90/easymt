from io import TextIOWrapper
import sys


def slice_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main(args):
    print("Starting: Chunking", file=sys.stderr)

    input_stream = TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    for i, line in enumerate(input_stream, start=1):
        chunks = slice_list(line.strip().split(), args.max_len)
        for sen in chunks:
            to_write = " ".join(sen)
            print(f"{to_write}", file=sys.stdout)

        if i % 100000 == 0:
            print(f"Processed lines: {i:,}", file=sys.stderr)

    print("Complete: Chunking", file=sys.stderr)
