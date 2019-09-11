import torch
from torch import nn
import torch.nn.functional as F
from data import MAX_LENGTH

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers=3, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.layers,
                          dropout=(0 if layers == 1 else dropout_p), bidirectional=True)

    def forward(self, input, input_lengths, hidden=None):
        print('debug in encoder, input size=%s'%input.size())
        input = input.transpose(0, 1)
        # Convert word indexes to embeddings
        embedded = self.embedding(input)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs.transpose(0,1), hidden.tranpose(0,1)

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, layers=3, dropout_p=0.1, attn_model='dot'):
        super(AttnDecoderRNN, self).__init__()

        # Keep for referencef
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, layers, dropout=(0 if layers == 1 else dropout_p))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input, last_hidden, encoder_outputs):
        input = input.transpose(0, 1)
        last_hidden = last_hidden.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0,1)
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        print('debug embedded size=', embedded.size())
        print('debug last_hidden size=', last_hidden.size())
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output.transpose(0,1), hidden.tranpose(0,1)
