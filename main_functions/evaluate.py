import os
import subprocess


def evaluate(args):
    os.system(
        "perl preprocessing-tools/multi-bleu-detok.perl "
        f" {args.reference} < {args.translation}"
    )

    print("-----\nSacreBLEU:")
    command = (
        f"sacrebleu {args.reference} -i {args.translation}"
    )
    result = subprocess.check_output(command, shell=True)
    result = eval(result)
    for key, value in result.items():
        print(f"{key:17}{value}")