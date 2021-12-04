import re


def name_suffix_from_file(filename):
    name = re.match(r"(.*)\.", filename).group(1)
    suffix = re.search(r"\.(.*)$", filename).group(1)
    return name, suffix
