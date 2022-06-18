import re


class NumReplacer:
    num_token = "<num>"
    number = re.compile(r"(?<=\s)\d[\d,'.]*\b")

    def __call__(self, line):
        clean = re.sub(self.number, self.num_token, line)
        return clean.strip()

    def __repr__(self):
        return "NumReplacer"