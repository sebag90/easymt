import os
import re


def split_filename(filename):
    """
    extracts name and suffix from a filename
    """
    # extract path
    split_path = filename.split(os.sep)
    path = os.sep.join(split_path[:-1])

    name = re.match(r"(.*)\.", split_path[-1])
    suffix = re.search(r"[^.]+$", split_path[-1])

    if name is not None and suffix is not None:
        f_name = name.group(1)
        suff = suffix.group()
        return path, f_name, suff

    # no suffix
    return path, name, ""


def count_lines(filename):
    """
    counts the line in a document
    """
    counter = 0
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            counter += 1

    return counter
