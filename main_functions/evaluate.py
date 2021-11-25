import os


def evaluate(args):
    os.system(
        "perl preprocessing-tools/multi-bleu-detok.perl "
        f" {args.reference} < {args.translation}"
    )

    print("-----")

    os.system(
        f"sacrebleu {args.reference} -i {args.translation}"
    )
