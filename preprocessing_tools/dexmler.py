import subprocess


class Dexmler:
    def __init__(self):
        self.args = [
            "perl",
            "preprocessing_tools/perl_scripts/de-xml.perl",
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
    t = Dexmler()
    print(t("<tag>good", "<tag>bien"))
