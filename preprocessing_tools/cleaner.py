import subprocess


class Cleaner:
    def __init__(self, n_min, n_max):
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/clean_corpus.perl",
            str(n_min),
            str(n_max)
        ]

    def __repr__(self):
        return f"Cleaner({self.language})"

    def __call__(self, line1, line2):
        result = subprocess.run(
            self.args, input=str.encode(f"{line1}\t{line2}\n"),
            capture_output=True
        )

        strings = result.stdout.decode("UTF-8").strip().split("\t")
        if len(strings) == 2:
            return strings
        else:
            return ["", ""]


if __name__ == "__main__":
    t = Cleaner(1, 5)
    print(t("something something", "c sjft e fhf f d d d d d d d d d d"))
    print(t("this is a good thins", "una buona cosa"))
