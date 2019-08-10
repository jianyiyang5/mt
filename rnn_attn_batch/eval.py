import torch
import os
import random
from data import MAX_LENGTH, SOS_token, EOS_token
from prepare_data import tensorFromSentence, indexesFromSentence2
from preprocess import prepareData
from model import EncoderRNN, AttnDecoderRNN
from decode import GreedySearchDecoder

def evaluate(device, searcher, input_voc, output_voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence2(input_voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [output_voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(device, searcher, input_voc, output_voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Evaluate sentence
            output_words = evaluate(device, searcher, input_voc, output_voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")

def evaluateRandomly(device, pairs, encoder, decoder, input_lang, output_lang, n=10):
    searcher = GreedySearchDecoder(encoder, decoder, device)
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(device, searcher, input_lang, output_lang, pair[0], max_length=MAX_LENGTH)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def decode(device, pairs, encoder, decoder, input_lang, output_lang):
    with open("test/test.hyp", "w", encoding='utf-8') as f:
        searcher = GreedySearchDecoder(encoder, decoder, device)
        for pair in pairs:
            output_words = evaluate(device, searcher, input_lang, output_lang, pair[0], max_length=MAX_LENGTH)
            output_sentence = ' '.join(output_words).replace('EOS', '').strip()
            f.write(output_sentence + '\n')

# def evaluate(device, encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentence, device)
#         input_length = input_tensor.size()[0]
#         encoder_hidden = encoder.initHidden()
#
#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                      encoder_hidden)
#             encoder_outputs[ei] += encoder_output[0, 0]
#
#         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
#
#         decoder_hidden = encoder_hidden[:encoder.layers, :, :] + encoder_hidden[encoder.layers:, :, :]
#
#         decoded_words = []
#         decoder_attentions = torch.zeros(max_length, max_length)
#
#         for di in range(max_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             decoder_attentions[di] = decoder_attention.data
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             else:
#                 decoded_words.append(output_lang.index2word[topi.item()])
#
#             decoder_input = topi.squeeze().detach()
#
#         return decoded_words, decoder_attentions[:di + 1]
#
# def evaluateRandomly(device, pairs, encoder, decoder, input_lang, output_lang, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(device, encoder, decoder, input_lang, output_lang, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')

def main():
    nIters = 25000
    loadFilename = os.path.join('checkpoints', '{}_{}.tar'.format(nIters, 'checkpoint'))
    checkpoint = torch.load(loadFilename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True, 'data')
    # If loading a model trained on GPU to CPU
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    hidden_size = 512
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0).to(device)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # encoder_optimizer_sd = checkpoint['en_opt']
    # decoder_optimizer_sd = checkpoint['de_opt']
    input_lang.__dict__ = checkpoint['input_lang']
    output_lang.__dict__ = checkpoint['output_lang']
    evaluateRandomly(device, pairs, encoder, decoder, input_lang, output_lang)
    _, _, test_pairs = prepareData('eng', 'fra', True, dir='test', filter=False)
    decode(device, test_pairs, encoder, decoder, input_lang, output_lang)

if __name__ == '__main__':
    main()
