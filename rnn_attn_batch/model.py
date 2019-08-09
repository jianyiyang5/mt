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
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded
        # output, hidden = self.gru(output, hidden)
        # output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # return output, hidden

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
        return outputs, hidden

    # def initHidden(self):
    #     return torch.zeros(self.layers*2, 1, self.hidden_size, device=self.device)

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, device, layers=3):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.device = device
#         self.layers = layers
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, self.layers)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(self.layers, 1, self.hidden_size, device=self.device)

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
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        # self.dropout_p = dropout_p
        # self.max_length = max_length
        # self.device = device
        # self.layers = layers
        #
        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.layers)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

        # Keep for reference
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
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        #
        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                          encoder_outputs.unsqueeze(0))
        #
        # output = torch.cat((embedded[0], attn_applied[0]), 1)
        # output = self.attn_combine(output).unsqueeze(0)
        #
        # output = F.relu(output)
        # output, hidden = self.gru(output, hidden)
        #
        # output = F.log_softmax(self.out(output[0]), dim=1)
        # return output, hidden, attn_weights

        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
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
        return output, hidden

    # def initHidden(self):
    #     return torch.zeros(self.layers, 1, self.hidden_size, device=self.device)

