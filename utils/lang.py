from functools import total_ordering

import torch


class Language:

    def __init__(self, name):
        self.name = name
        self.word2index = {
            "SOS": 1,
            "EOS": 2,
            "UNK": 3
        }
        self.index2word = {
            1: "SOS",
            2: "EOS",
            3: "UNK"
        }
        self.n_words = 4  # exclude SOS, EOS, UNK and PAD

    def __repr__(self):
        return f"Language({self.name})"

    def add_sentence(self, sentence):
        sentence = sentence.strip()
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            # add new word and update counters
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def read_vocabulary(self, vocfile):
        # read vocabulary from file
        with open(vocfile, "r", encoding="utf-8") as srcvoc:
            for line in srcvoc:
                word, count = line.strip().split("\t")
                self.add_word(word)

    def indexes_from_sentence(self, sentence):
        """
        convert a sentence of words into a
        sequence of indeces from the language
        if word is unknown use UNK vector
        """
        sentence = sentence.strip()
        unk = self.word2index["UNK"]
        sen = [
            self.word2index[word] if word in self.word2index else unk
            for word in sentence.split()
        ]

        # append EOS sentence
        sen.append(self.word2index["EOS"])
        return torch.tensor(sen)


class Vocab:
    def __init__(self, language, min_freq):
        self.language = language
        self.min_freq = min_freq
        self.voc = {}

    def add_sentence(self, sentence):
        sentence = sentence.strip()
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.voc:
            self.voc[word] = 0
        self.voc[word] += 1

    def save_voc(self):
        """
        sort by values (for eventual pruning)
        """
        outputfile = f"data/vocab.{self.language}"
        sorted_vocab = dict(
            reversed(
                sorted(
                    self.voc.items(), key=lambda item: item[1]
                )
            )
        )
        with open(outputfile, "w", encoding="utf-8") as ofile:
            for word, count in sorted_vocab.items():
                if count >= self.min_freq:
                    ofile.write(f"{word}\t{count}\n")


@total_ordering
class Hypothesis:
    def __init__(self):
        self.score = 0
        self.tokens = []
        self.decoder_state = None

    def __str__(self):
        return str(self.score)

    def __repr__(self):
        return self.__str__()

    def add_word(self, word):
        self.tokens.append(word)

    def __lt__(self, other):
        return self.weigthed_score < other.weigthed_score

    @property
    def weigthed_score(self):
        return self.score / len(self.tokens)

    @property
    def last_word(self):
        return self.tokens[-1]

    @property
    def sentence(self):
        return self.__str__()

    def get_indeces(self):
        return torch.tensor(self.tokens)
