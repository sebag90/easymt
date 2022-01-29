from model.rnn.seq2seq import seq2seq
from model.transformer.model import Transformer
from model.transformer.language_model import LanguageModel

from model.embedding_layer import EmbeddingLayer
from model.rnn.encoder import Encoder as rnn_encoder
from model.rnn.decoder import Decoder as rnn_decoder
from model.transformer.blocks import Encoder as tr_encoder
from model.transformer.blocks import Decoder as tr_decoder

from utils.errors import DimensionError


class ModelGenerator:
    def translation(self, params, src_language, tgt_language):
        if params.model.type == "rnn":
            encoder = rnn_encoder(
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

            embedding = EmbeddingLayer(
                src_lang=src_language,
                tgt_lang=tgt_language,
                word_vec_size=params.rnn.word_vec_size,
                shared=params.model.shared_embedding
            )

            model = seq2seq(
                encoder,
                decoder,
                embedding,
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

            embedding = EmbeddingLayer(
                src_lang=src_language,
                tgt_lang=tgt_language,
                word_vec_size=params.transformer.d_model,
                shared=params.model.shared_embedding
            )

            model = Transformer(
                encoder,
                decoder,
                embedding,
                src_language,
                tgt_language,
                params.model.max_length
            )

        return model

    def lm(self, params, language):
        if params.model.task == "language generation":
            # GPT style language model
            lm_type = "generator"

        elif params.model.task == "language encoding":
            # BERT style language model
            lm_type = "encoder"

        if not params.model.encoder_layers == params.model.decoder_layers:
            raise DimensionError(
                "In language models the number of layers in the "
                "encoder and decoder must match"
            )

        encoder = tr_encoder(
            d_model=params.transformer.d_model,
            n_head=params.transformer.heads,
            dim_ff=params.transformer.dim_feedforward,
            attn_dropout=params.transformer.attn_dropout,
            residual_dropout=params.transformer.residual_dropout,
            num_layers=params.model.encoder_layers,
            max_len=params.model.max_length
        )

        # force shared embedding
        embedding = EmbeddingLayer(
            src_lang=language,
            tgt_lang=language,
            word_vec_size=params.transformer.d_model,
            shared=True
        )

        model = LanguageModel(
            encoder,
            embedding,
            language,
            params.model.max_length,
            lm_type
        )

        return model


    def generate_model(self, params, src_language, tgt_language):
        if params.model.task == "translation":
            return self.translation(params, src_language, tgt_language)

        elif params.model.task in {"language generation", "language encoding"}:
            # für language generation we only use the source language
            return self.lm(params, src_language)
