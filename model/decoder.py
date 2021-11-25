import torch
import torch.nn as nn

from model.attention import Attention


class AttentionDecoder(nn.Module):
    def __init__(
            self,
            attn_model,
            word_vec_size,
            hidden_size,
            output_size,
            layers,
            dropout=0.1):
        super().__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout_p = dropout
        self.relu = nn.ReLU()

        # Define layers
        self.embedding = nn.Embedding(
            output_size,
            word_vec_size,
            padding_idx=0
        )
        self.rnn = nn.LSTM(
            word_vec_size + hidden_size,
            hidden_size,
            num_layers=layers,
            dropout=(0 if layers == 1 else dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attention(
                attn_model, hidden_size
            )

    def forward(
            self,
            input_step,
            context_vector,
            last_hidden,
            encoder_cell,
            encoder_outputs):
        # Get the embedding of the current input word
        embedded = self.embedding(input_step)

        rnn_input = torch.cat(
            (embedded, context_vector.unsqueeze(0)),
            2
        )

        # forward through RNN
        rnn_output, (hidden, cell) = self.rnn(
            rnn_input, (last_hidden, encoder_cell)
        )

        # calculate output with attention
        attention_output = self.attn(rnn_output, encoder_outputs)

        # pass through dense layer
        decoder_output = self.out(attention_output)
        decoder_output = self.dropout(decoder_output)

        return decoder_output, attention_output, hidden, cell
