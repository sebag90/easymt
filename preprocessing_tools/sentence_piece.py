import io
import pickle
import sentencepiece as spm
from pathlib import Path


class SentencePieceTokenizer:
    def __init__(self, size):
        self.size = size
        self.model = None
        self.processor = None
        self.trained = False

    def __repr__(self):
        return f"SentencePieceTokenizer({self.size})"

    def __call__(self, line):
        encoded = self.processor.encode(
            line.strip(), out_type=str
        )
        return ' '.join(encoded)

    def train(self, input_file, bpe, max_lines):
        model = io.BytesIO()
        #input_file.seek(0)

        # train and load model
        spm.SentencePieceTrainer.train(
            sentence_iterator=input_file,
            model_writer=model,
            vocab_size=self.size,
            bos_id=-1,
            eos_id=-1,
            unk_surface="<unk>",
            input_sentence_size=max_lines,
            model_type=("bpe" if bpe is True else "unigram")
        )
        self.model = model.getvalue()
        self.processor = spm.SentencePieceProcessor(
            model_proto=model.getvalue()
        )
        self.trained = True

    def save_model(self, outputpath):
        with Path(outputpath).open("wb") as ofile:
            ofile.write(self.model)

    @classmethod
    def from_pretrained(cls, model):
        data = model.read_bytes()
        model = spm.SentencePieceProcessor(
            model_proto=data
        )
        processor = cls(model.get_piece_size())

        processor.model = data
        processor.processor = model
        processor.trained = True
        return processor

    def decode(self, line):
        return self.processor.decode(line.split())

    def get_vocab(self):
        for id_n in range(self.processor.get_piece_size()):
            yield (
                self.processor.id_to_piece(id_n),
                round(self.processor.get_score(id_n), 4)
            )
