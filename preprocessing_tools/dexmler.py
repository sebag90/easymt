import re


class Dexmler:
    xml_tags = re.compile(r"<.+?>")
    spaces = re.compile(r"\s+")

    def __repr__(self):
        return f"Dexmler"

    def __call__(self, line1, line2):
        if line1.strip() == "" or line2.strip() == "":
            return ("", "")

        tags_1 = re.findall(self.xml_tags, line1)
        tags_2 = re.findall(self.xml_tags, line2)

        if len(tags_1) != len(tags_2):
            return ("", "")

        # remove xml tags
        cleaned1 = re.sub(self.xml_tags, "", line1)
        cleaned2 = re.sub(self.xml_tags, "", line2)

        # remove extra spaces
        norm_spaces1 = re.sub(self.spaces, " ", cleaned1)
        norm_spaces2 = re.sub(self.spaces, " ", cleaned2)

        return (norm_spaces1.strip(), norm_spaces2.strip())


if __name__ == "__main__":
    t = Dexmler()
    print(t("<tag>good", "<tag>bien"))
    print(t(
        "<this it> is some </b> nasty text <\5> xml-makrup </bien>",
        "<this it> is some </b> nasty text <\5> xml-makrup </bene>"
    ))
    print(t(
        "<this it> is some </b> nasty text xml-makrup ",
        "<this it> is some </b> nasty text xml-makrup "
    ))
