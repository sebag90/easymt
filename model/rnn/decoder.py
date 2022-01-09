import torch
import torch.nn as nn

from model.rnn.attention import Attention


class Decoder(nn.Module):
    def __init__(
            self,
            attn_model,
            word_vec_size,
            hidden_size,
            output_size,
            layers,
            rnn_dropout=0.3,
            attn_dropout=0.1,
            input_feed=True):
        super().__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.input_feed = input_feed

        if input_feed:
            rnn_input_size = word_vec_size + hidden_size
        else:
            rnn_input_size = word_vec_size

        # Define layers
        self.rnn = nn.LSTM(
            rnn_input_size,
            hidden_size,
            num_layers=layers,
            dropout=(rnn_dropout if layers > 1 else 0)
        )
        self.dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model.lower() != 'none':
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
        if self.input_feed:
            # concatenate current word input with previous context vector
            rnn_input = torch.cat(
                (input_step, context_vector.unsqueeze(0)),
                dim=2
            )
        else:
            rnn_input = input_step

        # forward through RNN
        rnn_output, (hidden, cell) = self.rnn(
            rnn_input, (last_hidden, encoder_cell)
        )

        if self.attn_model.lower() != "none":
            # calculate output with attention
            attention_output = self.attn(rnn_output, encoder_outputs)
            out_input = attention_output
        else:
            # no attention, pass the rnn output to out layer.
            # context vector is just a placeholder
            # of size (batch, hidden)
            out_input = rnn_output.squeeze(0)
            batch_size = input_step.shape[-1]
            attention_output = torch.zeros(
                batch_size, self.hidden_size
            )

        # pass through dense layer
        decoder_output = self.out(out_input)
        decoder_output = self.dropout(decoder_output)

        # Output dimensions:
        # decoder:      (batch, target_vocab)
        # attention:    (batch, hidden)
        # hidden, cell: (n layers, batch, hidden)
        return decoder_output, attention_output, hidden, cell
