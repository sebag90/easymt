import sys


def slice_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main(args):
    print("Starting: Chunking", file=sys.stderr)

    for i, line in enumerate(sys.stdin, start=1):
        chunks = slice_list(line.strip().split(), args.n)
        for sen in chunks:
            to_write = " ".join(sen)
            print(f"{to_write}\n", file=sys.stdout)

        if i % 100000 == 0:
            print(f"Processed lines: {i:,}", file=sys.stderr)

    print("Complete: Chunking", file=sys.stderr)
