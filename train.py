import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import json
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import get_loader, CocoDataset, collate_fn
from build_vocab import Vocabulary
from models import EncoderCNN, DecoderWithAttention
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Load vocabulary
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Training and validation data loaders
    train_loader = get_loader(args.image_dir, args.caption_path, vocab, transform,
                              args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(args.image_dir, args.val_caption_path, vocab, transform,
                            args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize encoder and decoder models
    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(attention_dim=512, embed_size=args.embed_size,
                                   hidden_size=args.hidden_size, vocab_size=len(vocab)).to(device)

    start_epoch = 0
    # Optionally resume training from a previous checkpoint (useful if training was interrupted or crashed)
    if args.resume_epoch > 0:
        encoder_ckpt = os.path.join(args.model_path, f'encoder-{args.resume_epoch}-647.ckpt')
        decoder_ckpt = os.path.join(args.model_path, f'decoder-{args.resume_epoch}-647.ckpt')
        if os.path.exists(encoder_ckpt) and os.path.exists(decoder_ckpt):
            encoder.load_state_dict(torch.load(encoder_ckpt))
            decoder.load_state_dict(torch.load(decoder_ckpt))
            print(f"Resumed from epoch {args.resume_epoch}")
            start_epoch = args.resume_epoch

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    params = list(decoder.parameters()) + list(filter(lambda p: p.requires_grad, encoder.parameters()))
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Load previous training history if exists
    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []

    history_path = os.path.join(args.model_path, 'loss_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            train_perplexities = history.get('train_perplexities', [])
            val_perplexities = history.get('val_perplexities', [])

    for epoch in range(start_epoch, args.num_epochs):
        decoder.train()
        encoder.train()
        running_loss = 0.0
        running_perplexity = 0.0
        total_step = len(train_loader)

        # -------- Training --------
        for i, (images, captions, lengths) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)
            decode_lengths = [l - 1 for l in lengths]
            targets = pack_padded_sequence(captions[:, 1:], decode_lengths, batch_first=True)[0]

            optimizer.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            outputs_packed = pack_padded_sequence(outputs, decode_lengths, batch_first=True)[0]
            loss = criterion(outputs_packed, targets)

            loss.backward()
            optimizer.step()

            perplexity = np.exp(loss.item())
            running_loss += loss.item()
            running_perplexity += perplexity

            if i % args.log_step == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i}/{total_step}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}')

        # Average training metrics
        avg_train_loss = running_loss / total_step
        avg_train_perplexity = running_perplexity / total_step
        train_losses.append(avg_train_loss)
        train_perplexities.append(avg_train_perplexity)

        # -------- Validation --------
        decoder.eval()
        encoder.eval()
        val_loss = 0.0
        val_perplexity = 0.0

        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images, captions = images.to(device), captions.to(device)
                decode_lengths = [l - 1 for l in lengths]
                targets = pack_padded_sequence(captions[:, 1:], decode_lengths, batch_first=True)[0]

                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                outputs_packed = pack_padded_sequence(outputs, decode_lengths, batch_first=True)[0]
                loss = criterion(outputs_packed, targets)

                val_loss += loss.item()
                val_perplexity += np.exp(loss.item())

        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_perplexity = val_perplexity / len(val_loader)
        val_losses.append(avg_val_loss)
        val_perplexities.append(avg_val_perplexity)

        print(f'>>> Epoch [{epoch+1}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'>>> Epoch [{epoch+1}], Train Perplexity: {avg_train_perplexity:.4f}, Val Perplexity: {avg_val_perplexity:.4f}')

        # Save model after each epoch
        torch.save(decoder.state_dict(), os.path.join(args.model_path, f'decoder-{epoch+1}-{i+1}.ckpt'))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, f'encoder-{epoch+1}-{i+1}.ckpt'))

        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_perplexities': train_perplexities,
            'val_perplexities': val_perplexities
        }
        with open(history_path, 'w') as f:
            json.dump(history, f)

    # -------- Plotting Results --------
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.model_path, 'loss_plot.png'))

    plt.figure(figsize=(10, 6))
    plt.plot(train_perplexities, label='Training Perplexity')
    plt.plot(val_perplexities, label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training & Validation Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.model_path, 'perplexity_plot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='Path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='Size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab_nostre.pkl', help='Path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='Directory for training images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='Path for train annotation file')
    parser.add_argument('--val_caption_path', type=str, default='data/annotations/captions_val2014.json', help='Path for validation annotation file')
    parser.add_argument('--log_step', type=int, default=1, help='Step size for printing log info')
    
    parser.add_argument('--embed_size', type=int, default=256, help='Dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='Dimension of LSTM hidden states')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume training from')

    args = parser.parse_args()
    print(args)
    main(args)
