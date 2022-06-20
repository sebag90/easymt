import io

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, size, max_lines):
        self.size = size
        self.model = None
        self.trained = False
        self.max_lines = max_lines

    def __repr__(self):
        return f"SentencePieceTokenizer({self.size})"

    def __call__(self, line):
        encoded = self.model.encode(
            line.strip(), out_type=str
        )
        return ' '.join(encoded)

    def train(self, input_file):
        model = io.BytesIO()
        input_file.seek(0)

        # train and load model
        spm.SentencePieceTrainer.train(
            sentence_iterator=input_file,
            model_writer=model,
            vocab_size=self.size,
            bos_id=-1,
            eos_id=-1,
            unk_surface="<unk>",
            input_sentence_size=self.max_lines
        )
        self.model = spm.SentencePieceProcessor(
            model_proto=model.getvalue()
        )
        self.trained = True

    def decode(self, line):
        return self.model.decode(line)

    def get_vocab(self):
        for id_n in range(self.model.get_piece_size()):
            yield (
                self.model.id_to_piece(id_n),
                round(self.model.get_score(id_n), 4)
            )
