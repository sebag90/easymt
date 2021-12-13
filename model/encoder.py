import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            word_vec_size,
            hidden_size,
            layers,
            rnn_dropout=0.3,
            bidirectional=True,
            decoder_layers=None):
        super().__init__()
        # properties
        self.hidden_size = hidden_size
        self.layers = layers
        self.bidirectional = bidirectional
        self.decoder_layers = decoder_layers
        self.layer_adaptaion = False

        # layers
        self.embedding = nn.Embedding(
            vocab_size,
            word_vec_size,
            padding_idx=0
        )
        self.rnn = nn.LSTM(
            word_vec_size,
            hidden_size,
            num_layers=layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional
        )

        # projection layers if bidirectional
        if bidirectional:
            self.hidden_direction_projection = nn.Linear(
                hidden_size * 2, hidden_size,
                bias=False
            )
            self.cell_direction_projection = nn.Linear(
                hidden_size * 2, hidden_size,
                bias=False
            )
            self.output_direction_projection = nn.Linear(
                hidden_size * 2, hidden_size,
                bias=False
            )

        # projection layers if encoder and decoder
        # do not have the same number of layers
        if decoder_layers is not None and decoder_layers != layers:
            self.layer_adaptaion = True
            self.hidden_layer_projection = nn.Linear(
                hidden_size * layers, hidden_size,
                bias=False
            )

            self.cell_layer_projection = nn.Linear(
                hidden_size * layers, hidden_size,
                bias=False
            )

    def forward(self, input_seq, input_lengths):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, enforce_sorted=False
        )

        # Forward pass through RNN
        outputs, (hidden_state, state_cell) = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        if self.bidirectional:
            # adjust tensor shapes of hidden_state, cell and output
            # if encoder is bidirectional
            hidden_state = hidden_state.view(
                self.layers, input_seq.shape[1], self.hidden_size * 2
            )

            hidden_state = self.hidden_direction_projection(hidden_state)

            state_cell = state_cell.view(
                self.layers, input_seq.shape[1], self.hidden_size * 2
            )
            state_cell = self.cell_direction_projection(state_cell)

            outputs = self.output_direction_projection(outputs)

        if self.layer_adaptaion:
            # adjust tensor shapes of hidden_state, cell and output
            # if encoder and decoder have different number of layers
            hidden_state = hidden_state.view(
                self.decoder_layers,
                input_seq.shape[1],
                self.hidden_size * self.layers
            )
            hidden_state = self.hidden_layer_projection(hidden_state)

            state_cell = state_cell.view(
                self.decoder_layers,
                input_seq.shape[1],
                self.hidden_size * self.layers
            )
            state_cell = self.cell_layer_projection(state_cell)

        # Output dimensions:
        # output:       (length, batch, hidden_size)
        # cell, state:  (n_layers, batch, hidden_size)
        return outputs, hidden_state, state_cell
