from pathlib import Path
import re

from utils.utils import split_filename


def main(args):
    reference = Path(args.reference)
    translation = Path(args.translation)

    # compile overly complicated number regex
    number = re.compile(r"(?<=\s)\d[\d,'.]*\b")

    path, name, suffix = split_filename(args.translation)
    ofile = Path(f"{path}/{name}.numbered.{suffix}").open("w")

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

            ofile.write(f"{line}\n")
            if (i+1) % 100000 == 0:
                print(f"Processed lines: {i + 1:,}", flush=True)

    ofile.close()

    print("Complete: Replacing numbers")
