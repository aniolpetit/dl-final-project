import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms, models
from data_loader import get_loader
from build_vocab import Vocabulary
import matplotlib.pyplot as plt
from models import EncoderCNN, Attention, DecoderWithAttention
from torch.utils.data import Subset
import random
from data_loader import collate_fn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# Training Script
# -----------------------
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image transforms
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

    dataset = get_loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size,
                     shuffle=True, num_workers=args.num_workers, return_dataset=True)

    # Use a subset of the dataset (e.g., first 3000 items)
    subset_size = 50000
    indices = random.sample(range(len(dataset)), subset_size)
    subset_dataset = Subset(dataset, indices)

    # Now create data loader with the subset
    data_loader = torch.utils.data.DataLoader(
        dataset=subset_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn  # assuming your dataset has a collate_fn
    )

    # Model components
    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(attention_dim=512, embed_size=args.embed_size,
                                   hidden_size=args.hidden_size, vocab_size=len(vocab)).to(device)

    pad_idx = vocab('<pad>')  # make sure this matches your <pad> token
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(filter(lambda p: p.requires_grad, encoder.parameters()))
    optimizer = optim.Adam(params, lr=args.learning_rate)

    total_step = len(data_loader)
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []

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

        # ---------------- Validation ----------------
        decoder.eval()
        encoder.eval()
        val_loss = 0.0
        val_perplexity = 0.0
        with torch.no_grad():
            for images, captions, lengths in data_loader:  # (Use same data_loader for simplicity; ideally use a separate val_loader)
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

    # ---------------- Plotting ----------------
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


# -----------------------
# Argparse
# -----------------------
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

    args = parser.parse_args()
    print(args)
    main(args)
