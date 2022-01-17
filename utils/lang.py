from functools import total_ordering

import torch


class Language:
    def __init__(self, name):
        self.name = name
        self.word2index = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>": 3
        }
        self.index2word = {
            0: "<pad>",
            1: "<sos>",
            2: "<eos>",
            3: "<unk>"
        }

    def __repr__(self):
        return f"Language({self.name})"

    def __len__(self):
        assert len(self.word2index) == len(self.index2word)
        return len(self.index2word)

    def add_sentence(self, sentence):
        sentence = sentence.strip()
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            # add new word and update counters
            idx = len(self)
            self.word2index[word] = idx
            self.index2word[idx] = word

    def read_vocabulary(self, vocfile):
        """
        read vocabulary from TSV file
        the first element will be considered as word
        the rest will be gnored
        """
        with open(vocfile, "r", encoding="utf-8") as srcvoc:
            for line in srcvoc:
                word, *rest = line.strip().split("\t")
                self.add_word(word)

    def toks2idx(self, tokens):
        """
        convert a list of tokens into a
        sequence of indeces from the language
        if word is unknown use <unk> vector
        """
        unk = self.word2index["<unk>"]
        sen = [
            self.word2index[word] if word in self.word2index else unk
            for word in tokens
        ]

        # append <eos> sentence
        sen.append(self.word2index["<eos>"])
        return torch.tensor(sen)

    def idx2toks(self, indeces):
        return [self.index2word[i] for i in indeces]


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

    def save_voc(self, path):
        """
        sort by values (for eventual pruning)
        """
        outputfile = f"{path}/vocab.{self.language}"
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
    def __init__(self, alpha=2):
        self.score = 0
        self.tokens = list()
        self.attention_stack = list()
        self.decoder_state = None
        self.doubles = dict()
        self.alpha = alpha

    def __str__(self):
        return str(self.weigthed_score)

    def __repr__(self):
        return f"Hypothesis({round(self.weigthed_score, 5)})"

    def __lt__(self, other):
        return self.weigthed_score < other.weigthed_score

    def update(self, word, decoder_state, log_prob):
        self.tokens.append(word)
        self.score += log_prob
        self.decoder_state = decoder_state
        self.attention_stack.append(
            decoder_state[0]  # attention vector
        )

        if self.alpha != 0:
            self.apply_weighting()

    def apply_weighting(self):
        """
        apply some weighting to the hypothesis score:
          - each repeating pattern in the hypothesis
            (ex. this pattern repeats, this pattern repeats)
            will influence the score by alpha * number of repetition
        """
        doubles = self.find_subs(self.get_indeces().tolist())
        if doubles:
            for key, value in doubles.items():
                if key not in self.doubles:
                    self.doubles[key] = value
                    extra_cost = value * self.alpha
                    self.score -= extra_cost

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

    def get_attention(self):
        return torch.cat(self.attention_stack)

    @staticmethod
    def pattern_match(sublist, complete_list):
        counter = 0
        for i in range(len(complete_list) - len(sublist) + 1):
            if complete_list[i:i+len(sublist)] == sublist:
                counter += 1
        return counter

    @staticmethod
    def find_subs(complete_list):
        results = dict()
        for i in range(len(complete_list)):
            for j in range(len(complete_list)):
                sub = complete_list[i:j]
                if len(sub) > 1:
                    n = Hypothesis.pattern_match(sub, complete_list)
                    if n > 1:
                        if tuple(sub) not in results:
                            results[tuple(sub)] = n

        return results
