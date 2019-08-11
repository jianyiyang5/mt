import torch
import os
import random
from data import MAX_LENGTH, SOS_token, EOS_token, Lang
from prepare_data import tensorFromSentence, indexesFromSentence2
from preprocess import prepareData
from model import EncoderRNN, AttnDecoderRNN
from decode import GreedySearchDecoder, GreedySearchDecoderBatch

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

def evaluate_batch(device, searcher, input_voc, output_voc, sentences, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence2(input_voc, sentence) for sentence in sentences]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    all_tokens, all_scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [[output_voc.index2word[token.item()] for token in tokens] for tokens in all_tokens]
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
    with open("test/test.ref", "w", encoding='utf-8') as f:
        for pair in pairs:
            f.write(pair[1] + '\n')

    with open("test/test.hyp", "w", encoding='utf-8') as f:
        searcher = GreedySearchDecoder(encoder, decoder, device)
        for pair in pairs:
            output_words = evaluate(device, searcher, input_lang, output_lang, pair[0], max_length=MAX_LENGTH)
            output_sentence = ' '.join(output_words).replace('EOS', '').strip()
            f.write(output_sentence + '\n')

def decode_batch(device, pairs, encoder, decoder, input_lang, output_lang, batch_size=64):
    with open("test/test.ref", "w", encoding='utf-8') as f:
        for pair in pairs:
            f.write(pair[1] + '\n')

    with open("test/test.hyp", "w", encoding='utf-8') as f:
        searcher = GreedySearchDecoderBatch(encoder, decoder, device)
        i = 0
        while i < len(pairs):
            sentences = [pairs[j][0] for j in range(i, min(i+batch_size, len(pairs)))]
            all_output_words = evaluate_batch(device, searcher, input_lang, output_lang, sentences, max_length=MAX_LENGTH)
            for output_words in all_output_words:
                output_sentence = ' '.join(output_words).replace('EOS', '').strip()
                f.write(output_sentence + '\n')
            i += batch_size

def main():
    nIters = 50000
    loadFilename = os.path.join('checkpoints', '{}_{}.tar'.format(nIters, 'checkpoint'))
    checkpoint = torch.load(loadFilename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input_lang, output_lang, pairs = prepareData('eng', 'fra', True, 'data', filter=False)
    # If loading a model trained on GPU to CPU
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    hidden_size = 512
    input_lang = Lang('fra')
    output_lang = Lang('eng')
    input_lang.__dict__ = checkpoint['input_lang']
    output_lang.__dict__ = checkpoint['output_lang']
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0).to(device)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # encoder_optimizer_sd = checkpoint['en_opt']
    # decoder_optimizer_sd = checkpoint['de_opt']
    _, _, test_pairs = prepareData('eng', 'fra', True, dir='test', filter=False)
    evaluateRandomly(device, test_pairs, encoder, decoder, input_lang, output_lang)
    decode_batch(device, test_pairs, encoder, decoder, input_lang, output_lang, batch_size=64)

if __name__ == '__main__':
    main()
