import torch
from torch import nn
from data import SOS_token, EOS_token

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

class GreedySearchDecoderBatch(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(GreedySearchDecoderBatch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seqs, input_lengths, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.layers]
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(input_seqs.size()[1])]])
        decoder_input = decoder_input.to(self.device)
        eos = torch.LongTensor([[EOS_token for _ in range(input_seqs.size()[1])]]).to(self.device)
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([input_seqs.size()[1],0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([input_seqs.size()[1],0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            if torch.equal(decoder_input, eos):
                break
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input.unsqueeze(1)), dim=1)
            all_scores = torch.cat((all_scores, decoder_scores.unsqueeze(1)), dim=1)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
