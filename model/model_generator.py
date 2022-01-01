from model.rnn.seq2seq import seq2seq
from model.transformer.model import Transformer

from model.rnn.encoder import Encoder as rnn_encoder
from model.rnn.decoder import Decoder as rnn_decoder
from model.transformer.blocks import Encoder as tr_encoder
from model.transformer.blocks import Decoder as tr_decoder


class ModelGenerator:
    def __init__(self, model_type):
        self.model_type = model_type

    def generate_model(self, params, src_language, tgt_language):
        if params.model.type == "rnn":
            encoder = rnn_encoder(
                vocab_size=len(src_language),
                word_vec_size=params.rnn.word_vec_size,
                hidden_size=params.rnn.hidden_size,
                layers=params.model.encoder_layers,
                rnn_dropout=params.rnn.rnn_dropout,
                bidirectional=params.rnn.bidirectional,
                decoder_layers=params.model.decoder_layers
            )

            decoder = rnn_decoder(
                attn_model=params.rnn.attention,
                word_vec_size=params.rnn.word_vec_size,
                hidden_size=params.rnn.hidden_size,
                output_size=len(tgt_language),
                layers=params.model.decoder_layers,
                rnn_dropout=params.rnn.rnn_dropout,
                attn_dropout=params.rnn.attn_dropout,
                input_feed=params.rnn.input_feed
            )

            model = seq2seq(
                encoder,
                decoder,
                src_language,
                tgt_language,
                params.model.max_length
            )

        elif params.model.type == "transformer":
            encoder = tr_encoder(
                d_model=params.transformer.d_model,
                n_head=params.transformer.heads,
                dim_ff=params.transformer.dim_feedforward,
                attn_dropout=params.transformer.attn_dropout,
                residual_dropout=params.transformer.residual_dropout,
                num_layers=params.model.encoder_layers,
                vocab_size=len(src_language),
                max_len=params.model.max_length
            )

            decoder = tr_decoder(
                d_model=params.transformer.d_model,
                n_head=params.transformer.heads,
                dim_ff=params.transformer.dim_feedforward,
                attn_dropout=params.transformer.attn_dropout,
                residual_dropout=params.transformer.residual_dropout,
                num_layers=params.model.decoder_layers,
                vocab_size=len(tgt_language),
                max_len=params.model.max_length
            )

            model = Transformer(
                encoder,
                decoder,
                src_language,
                tgt_language,
                params.model.max_length
            )

        return model
