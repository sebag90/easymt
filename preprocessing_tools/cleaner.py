import re


class Cleaner:
    def __init__(self, n_min=1, n_max=50, ratio=9):
        self.n_min = n_min
        self.n_max = n_max
        self.ratio = ratio
        self.spaces = re.compile(r"\s+")

    def __repr__(self):
        return f"Cleaner"

    def __call__(self, line1, line2):
        line1 = line1.strip()
        line2 = line2.strip()

        # ignore empty strings
        if line1 == "" or line2 == "":
            return ("", "")

        # make sure both sentences are within boundaries
        toks_1 = line1.split()
        toks_2 = line2.split()

        # enforce max length
        if len(toks_1) > self.n_max or len(toks_2) > self.n_max:
            return ("", "")

        # enforce min length
        if len(toks_1) < self.n_min or len(toks_2) < self.n_min:
            return ("", "")

        # enforce length raio between sentences
        if (len(toks_1) / len(toks_2) > self.ratio or
                len(toks_2)/len(toks_1) > self.ratio):
            return ("", "")

        # return lines
        return(line1, line2)


if __name__ == "__main__":
    t = Cleaner(1, 5)
    print(t("something something", "c sjft e fhf f d d d d d d d d d d"))
    print(t("this is a good thins", "una buona cosa"))
