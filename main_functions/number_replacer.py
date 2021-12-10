from pathlib import Path
import re


def replace_numbers(args):
    reference = Path(args.reference)
    translation = Path(args.translation)

    # compile overly complicated number regex
    number = re.compile(r"\b\d[\d,'.]*\b")

    ofile = open(Path("data/translation.numbered"), "w")

    with open(reference, "r", encoding="utf-8") as r_file, \
            open(translation, "r", encoding="utf-8") as t_file:
        for i, (line_r, line_t) in enumerate(zip(r_file, t_file)):
            # extract numbers from reference
            original_numbers = re.findall(number, line_r)

            # get <num> placeholders from translation
            place_holder = re.findall(r"<num>", line_t)

            if len(original_numbers) > 0:
                if len(original_numbers) == len(place_holder):
                    sen = list()
                    i = 0
                    for token in line_t.split():
                        if "<num>" in token:
                            sen.append(original_numbers[i])
                            i += 1
                        else:
                            sen.append(token)

            else:
                # no numbers or numbers don't match
                # leave translation as it is
                sen = line_t.strip().split()

            line = " ".join(sen)

            ofile.write(f"{line}\n")
            print(f"Replacing numbers: line {i}", end="\r")

    ofile.close()
    print(" " * 50, end="\r")
    print("Replacing numbers: complete")
