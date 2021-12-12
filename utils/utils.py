import re


def name_suffix_from_file(filename):
    """
    extracts name and suffix from a filename
    """
    name = re.match(r"(.*)\.", filename)
    suffix = re.search(r"[^.]+$", filename)
    if name is not None and suffix is not None:
        return name.group(1), suffix.group()

    return filename, ""


def count_lines(filename):
    """
    counts the line in a document
    """
    counter = 0
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            counter += 1

    return counter
