import torch
from torch import nn, optim
import random
import time
import os
from data import MAX_LENGTH, SOS_token, EOS_token
from prepare_data import tensorsFromPair, batch2TrainData
from preprocess import prepareData
from helper import timeSince
from plot import showPlot
from model import EncoderRNN, AttnDecoderRNN

teacher_forcing_ratio = 0.9

def maskNLLLoss(inp, target, mask, device):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(device, input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    input_variable = input_variable.transpose(0, 1)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    print('debug 40 input_variable size=', input_variable.size())
    print('debug 40 lengths size=', lengths.size())
    curMaxLen = torch.max(lengths)
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, device, curMaxLen)
    print('debug encoder_outputs size=', encoder_outputs.size())
    print('debug encoder_hidden size=', encoder_hidden.size())

    encoder_outputs = encoder_outputs.transponse(0,1)
    encoder_hidden = encoder_hidden.transpose(0,1)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.module.layers]
    # decoder_hidden = encoder_hidden[:decoder.layers]
    print('debug encoder_hidden size=%s, decoder_hidden size=%s, batch_size=%s'%(encoder_hidden.size(), decoder_hidden.size(), batch_size))

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input.transpose(0, 1), decoder_hidden.transpose(0, 1), encoder_outputs.transpose(0,1)
            )
            decoder_output = decoder_output.transpose(0,1)
            decoder_hidden = decoder_hidden.transpose(0,1)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input.transpose(0, 1), decoder_hidden.transpose(0, 1), encoder_outputs.transpose(0,1)
            )
            decoder_output = decoder_output.transpose(0, 1)
            decoder_hidden = decoder_hidden.transpose(0, 1)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(device, pairs, input_lang, output_lang, encoder, decoder, batch_size, n_iters, print_every=250, plot_every=250, save_every=2000, learning_rate=0.01, save_dir='checkpoints'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang, device)
                      for i in range(n_iters)]
    training_batches = [batch2TrainData(input_lang, output_lang, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iters)]

    for iter in range(1, n_iters + 1):
        training_batch = training_batches[iter - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(device, input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        # Save checkpoint
        if iter % save_every == 0:
            directory = save_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iter,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'input_lang': input_lang.__dict__,
                'output_lang': output_lang.__dict__
            }, os.path.join(directory, '{}_{}.tar'.format(iter, 'checkpoint')))

    showPlot(plot_losses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True, dir='data', filter=False)
    hidden_size = 512
    batch_size = 64
    iters = 50000
    # encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        attn_decoder = nn.DataParallel(attn_decoder)
    encoder = encoder.to(device)
    attn_decoder = attn_decoder.to(device)

    # attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(device, pairs, input_lang, output_lang, encoder, attn_decoder, batch_size, iters, print_every=250)

if __name__ == '__main__':
    main()

