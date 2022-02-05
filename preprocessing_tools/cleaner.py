import re


class Cleaner:
    def __init__(self, min_len=1, max_len=70, ratio=9):
        self.min_len = min_len
        self.max_len = max_len
        self.ratio = ratio
        self.spaces = re.compile(r"\s+")

    def __repr__(self):
        return "Cleaner"

    def process_two(self, line1, line2):
        line1 = line1.strip()
        line2 = line2.strip()

        # ignore empty strings
        if line1 == "" or line2 == "":
            return ("", "")

        # make sure both sentences are within boundaries
        toks_1 = line1.split()
        toks_2 = line2.split()

        # enforce max length
        if len(toks_1) > self.max_len or len(toks_2) > self.max_len:
            return ("", "")

        # enforce min length
        if len(toks_1) < self.min_len or len(toks_2) < self.min_len:
            return ("", "")

        # enforce length raio between sentences
        if (len(toks_1) / len(toks_2) > self.ratio or
                len(toks_2)/len(toks_1) > self.ratio):
            return ("", "")

        # return lines
        return (line1, line2)

    def process_single(self, line):
        line = line.strip()

        if line == "":
            return tuple([line])

        toks = line.split()

        # enforce max len
        if len(toks) > self.max_len:
            return tuple([""])

        # enforce min len
        if len(toks) < self.min_len:
            return tuple([""])

        return tuple([line])

    def __call__(self, *args):
        if len(args) == 1:
            return self.process_single(*args)
        else:
            return self.process_two(*args)


if __name__ == "__main__":
    t = Cleaner(1, 5)
    print(t("something something", "c sjft e fhf f d d d d d d d d d d"))
    print(t("this is a good thins", "una buona cosa"))
    print(t("this is a good thins"))
