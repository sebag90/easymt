import torch.nn as nn


class Projector(nn.Module):
    """
    The projector adjust the shape of the output of the
    encoder if:
        - encoder is bidirectional
        - encoder and decoder have different number of layers

    if none of the previous conditions apply, the projector
    simply returns the tensors without touching them
    """
    def __init__(
            self,
            bidirectional,
            hidden_size,
            encoder_layers,
            decoder_layers):
        super().__init__()
        self.direction = False
        self.layer = False
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size

        # projection layer for direction
        if bidirectional:
            self.direction = True
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

        # projection layers for layers mismatch
        if decoder_layers is not None and decoder_layers != encoder_layers:
            self.layer = True
            self.hidden_layer_projection = nn.Linear(
                hidden_size * encoder_layers, hidden_size * decoder_layers,
                bias=False
            )

            self.cell_layer_projection = nn.Linear(
                hidden_size * encoder_layers, hidden_size * decoder_layers,
                bias=False
            )

    def forward(self, outputs, hidden_state, state_cell, batch_size):
        if self.direction:
            # adapt tensor shapes of hidden_state, cell and output
            # if encoder is bidirectional
            hidden_state = self.hidden_direction_projection(
                hidden_state.view(
                    self.encoder_layers,
                    batch_size,
                    self.hidden_size * 2
                )
            )

            state_cell = self.cell_direction_projection(
                state_cell.view(
                    self.encoder_layers,
                    batch_size,
                    self.hidden_size * 2
                )
            )

            outputs = self.output_direction_projection(outputs)

        if self.layer:
            # only adapt hidden_state and state_cell to match
            # number of layers in the decoder
            hidden_state = self.hidden_layer_projection(
                hidden_state.view(
                    batch_size,
                    self.hidden_size * self.encoder_layers
                )
            )

            state_cell = self.cell_layer_projection(
                state_cell.view(
                    batch_size,
                    self.hidden_size * self.encoder_layers
                )
            )

            # adjust shapes
            hidden_state = hidden_state.view(
                self.decoder_layers, batch_size, self.hidden_size
            )
            state_cell = state_cell.view(
                self.decoder_layers, batch_size, self.hidden_size
            )

        return outputs, hidden_state, state_cell


class Encoder(nn.Module):
    def __init__(self,
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

        self.rnn = nn.LSTM(
            word_vec_size,
            hidden_size,
            num_layers=layers,
            dropout=(rnn_dropout if layers > 1 else 0),
            bidirectional=bidirectional
        )

        self.projector = Projector(
            bidirectional, hidden_size, layers, decoder_layers
        )

    def forward(self, input_seq, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            input_seq, input_lengths, enforce_sorted=False
        )

        # Forward pass through RNN
        outputs, (hidden_state, state_cell) = self.rnn(packed)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # pass through projector
        outputs, hidden_state, state_cell = self.projector(
            outputs, hidden_state, state_cell, input_seq.shape[1]
        )

        # Output dimensions:
        # output:       (length, batch, hidden_size)
        # cell, state:  (n_layers, batch, hidden_size)
        return outputs, hidden_state, state_cell
