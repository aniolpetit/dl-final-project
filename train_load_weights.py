import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import get_loader, collate_fn
import random
import matplotlib.pyplot as plt
from build_vocab import Vocabulary

from models import EncoderCNN, DecoderWithAttention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_old_decoder_weights(new_decoder, old_decoder_path, device):
    """
    Loads weights from the old DecoderRNN into the new DecoderWithAttention where possible.
    Assumes old_decoder_path points to a checkpoint of old DecoderRNN.

    The mapping is:
    - embedding layer weights
    - fc layer weights (linear layer after LSTM)
    - LSTM weights/biases can be copied into LSTMCell weights partially (not trivial, may skip)
    """

    old_state = torch.load(old_decoder_path, map_location=device)

    new_state = new_decoder.state_dict()

    # 1. Copy embedding weights if shape matches
    if 'embed.weight' in old_state and new_state.get('embedding.weight', None) is not None:
        if old_state['embed.weight'].shape == new_state['embedding.weight'].shape:
            new_state['embedding.weight'] = old_state['embed.weight']
            print('Loaded embedding weights from old model.')

    # 2. Copy linear output layer weights if shapes compatible
    # old: linear.weight, linear.bias
    # new: fc.weight, fc.bias
    if 'linear.weight' in old_state and 'linear.bias' in old_state:
        if old_state['linear.weight'].shape == new_state['fc.weight'].shape and \
           old_state['linear.bias'].shape == new_state['fc.bias'].shape:
            new_state['fc.weight'] = old_state['linear.weight']
            new_state['fc.bias'] = old_state['linear.bias']
            print('Loaded linear output layer weights from old model.')

    # 3. Skip LSTM weights mapping — because old model has nn.LSTM, new has nn.LSTMCell
    # (different param names, shape, structure). 
    # Implementing a precise mapping is complex and error-prone. You can choose to:
    # - skip loading LSTM weights, or
    # - write custom mapping if you want (advanced).

    # Load the updated weights into new decoder
    new_decoder.load_state_dict(new_state)

    print('Finished loading old decoder weights where possible.')


def freeze_encoder_and_decoder_except_attention(encoder, decoder):
    """
    Freeze encoder parameters and decoder parameters except the attention module
    """
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    # Freeze decoder except attention module parameters
    for name, param in decoder.named_parameters():
        if 'attention' in name or 'f_beta' in name:  # attention layers and gating
            param.requires_grad = True
        else:
            param.requires_grad = False

    print('Frozen encoder and decoder except attention and gating layers.')


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Transformations
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Load vocab
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    dataset = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size,
                        shuffle=True, num_workers=args.num_workers, return_dataset=True)

    # subset_size = 50000
    # indices = random.sample(range(len(dataset)), subset_size)
    # subset_dataset = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(attention_dim=512, embed_size=args.embed_size,
                                   hidden_size=args.hidden_size, vocab_size=len(vocab)).to(device)

    # Load old decoder weights
    if args.old_decoder_path:
        print(f'Loading old decoder weights from {args.old_decoder_path}')
        load_old_decoder_weights(decoder, args.old_decoder_path, device)

    # Freeze parts except attention
    freeze_encoder_and_decoder_except_attention(encoder, decoder)

    pad_idx = vocab('<pad>')
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Only params with requires_grad=True will be optimized (attention layers + gating)
    params = filter(lambda p: p.requires_grad, list(decoder.parameters()) + list(encoder.parameters()))
    optimizer = optim.Adam(params, lr=args.learning_rate)

    total_step = len(data_loader)
    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []

    for epoch in range(args.num_epochs):
        decoder.train()
        encoder.train()
        running_loss = 0.0
        running_perplexity = 0.0

        for i, (images, captions, lengths) in enumerate(data_loader):
            images, captions = images.to(device), captions.to(device)
            targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            outputs = pack_padded_sequence(outputs, [l - 1 for l in lengths], batch_first=True)[0]

            loss = criterion(outputs, targets)
            perplexity = np.exp(loss.item())

            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_perplexity += perplexity

            if i % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i}/{total_step}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {perplexity:5.4f}')

            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(args.model_path, f'decoder-{epoch+1}-{i+1}.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(args.model_path, f'encoder-{epoch+1}-{i+1}.ckpt'))

        avg_train_loss = running_loss / total_step
        avg_train_perplexity = running_perplexity / total_step
        train_losses.append(avg_train_loss)
        train_perplexities.append(avg_train_perplexity)

        # Validation (using same data_loader for simplicity)
        decoder.eval()
        encoder.eval()
        val_loss = 0.0
        val_perplexity = 0.0
        with torch.no_grad():
            for images, captions, lengths in data_loader:
                images, captions = images.to(device), captions.to(device)
                targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True)[0]

                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                outputs = pack_padded_sequence(outputs, [l - 1 for l in lengths], batch_first=True)[0]

                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_perplexity += np.exp(loss.item())

        avg_val_loss = val_loss / len(data_loader)
        avg_val_perplexity = val_perplexity / len(data_loader)
        val_losses.append(avg_val_loss)
        val_perplexities.append(avg_val_perplexity)

        print(f'>>> Epoch [{epoch+1}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'>>> Epoch [{epoch+1}], Train Perplexity: {avg_train_perplexity:.4f}, Val Perplexity: {avg_val_perplexity:.4f}')

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.model_path, 'loss_plot.png'))
    plt.show()

    # Plot perplexities
    plt.figure(figsize=(10, 6))
    plt.plot(train_perplexities, label='Training Perplexity')
    plt.plot(val_perplexities, label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training & Validation Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.model_path, 'perplexity_plot.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='Path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='Size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_nostre.pkl', help='Path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='Directory for training images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='Path for train annotation file')
    parser.add_argument('--log_step', type=int, default=10, help='Step size for printing log info')
    parser.add_argument('--save_step', type=int, default=3125, help='Step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='Dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='Dimension of LSTM hidden states')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # New argument: path to old decoder weights
    parser.add_argument('--old_decoder_path', type=str, default='models/decoder-5-3000.pkl', help='Path to old decoder weights to initialize')

    args = parser.parse_args()
    print(args)
    main(args)
