import re


def name_suffix_from_file(filename):
    name = re.match(r"(.*)\.", filename).group(1)
    suffix = re.search(r"[^.]+$", filename).group()
    return name, suffix


def count_lines(filename):
    counter = 0
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            counter += 1

    return counter
