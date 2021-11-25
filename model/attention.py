import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.linear_in = nn.Linear(
                hidden_size, hidden_size, bias=False
            )

        self.linear_out = nn.Linear(
            hidden_size * 2, hidden_size, bias=False
        )

    def score(self, decoder_outputs, encoder_outputs):
        if self.method == 'dot':
            decoder_outputs = decoder_outputs.permute(
                1, 0, 2  # (batch, 1, hidden)
            )
            encoder_outputs = encoder_outputs.permute(
                1, 0, 2  # (batch, max_len, hidden)
            )
            score = torch.bmm(
                encoder_outputs,
                decoder_outputs.transpose(1, 2)
            )   # (batch, max Len, 1)
            return score

        elif self.method == 'general':
            energy = self.linear_in(encoder_outputs)

            # transpose batch first for bmm
            decoder_outputs = decoder_outputs.permute(
                1, 0, 2  # (batch, 1, hidden)
            )
            energy = energy.permute(
                1, 0, 2  # (batch, max_len, hidden)
            )
            score = torch.bmm(
                energy,
                decoder_outputs.transpose(1, 2)
            )   # (batch, max Len, 1)
            return score

    def forward(self, decoder_outputs, encoder_outputs):
        # calculate alignment scores
        alignments = self.score(decoder_outputs, encoder_outputs)
        align_vector = F.softmax(alignments, dim=1)

        # reshape
        align_vector = align_vector.permute(0, 2, 1)

        # calculate context vector
        context = torch.bmm(
            align_vector, encoder_outputs.transpose(0, 1)
        )

        # concatenate context vector and decoder output
        # 1 - reshape
        context = context.squeeze(1)
        decoder_outputs = decoder_outputs.squeeze(0)

        # 2 -concatenate
        concat = torch.cat((context, decoder_outputs), dim=1)
        output = self.linear_out(concat)

        output = torch.tanh(output)
        return output
