import argparse
from pathlib import Path
import re
import sys


def main(args):
    reference = Path(args.reference)
    translation = Path(args.translation)

    if args.output is not None:
        output_file = Path(args.output).open("w", encoding="utf-8")
    else:
        output_file = sys.stdout

    # compile overly complicated number regex
    number = re.compile(r"(?<=\s)\d[\d,'.]*\b")

    with reference.open("r", encoding="utf-8") as r_file, \
            translation.open("r", encoding="utf-8") as t_file:
        for i, (line_r, line_t) in enumerate(zip(r_file, t_file)):
            # extract numbers from reference
            original_numbers = re.findall(number, line_r)

            # get <num> placeholders from translation
            place_holder = re.findall(r"<num>", line_t)

            if (len(original_numbers) > 0 and
                    len(original_numbers) == len(place_holder)):
                sen = list()
                i = 0
                for token in line_t.split():
                    if "<num>" in token:
                        # replace <num> with real number
                        numbered = token.replace(
                            "<num>", original_numbers[i]
                        )
                        sen.append(numbered)
                        i += 1
                    else:
                        sen.append(token)

            else:
                # no numbers or numbers don't match
                # leave translation as it is
                sen = line_t.strip().split()

            line = " ".join(sen)
            print(line, file=output_file)

            # print progress
            if i % 1000000 == 0:
                print(i, end="", file=sys.stderr, flush=True)

            elif i % 100000 == 0:
                print(".", end="", file=sys.stderr, flush=True)

    output_file.close()

    print("Complete: Replacing numbers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "replace <num> tokens in a translated file based on the "
            "reference (or source) file. <num> tokens will only be "
            "replaced if the script finds the same ammount of numbers "
            "between the reference and translated file"
        )
    )
    parser.add_argument(
        "--reference",
        action="store",
        help="path to reference translation"
    )
    parser.add_argument(
        "--translation",
        action="store",
        help="path to translated document"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        action="store",
    )

    args = parser.parse_args()
    main(args)
