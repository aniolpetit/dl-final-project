import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import pickle
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io

from models import EncoderCNN, DecoderWithAttention
from build_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Config
ENCODER_PATH = 'models/encoder-5-647.ckpt'
DECODER_PATH = 'models/decoder-5-647.ckpt'
VOCAB_PATH = 'data/vocab_nostre.pkl'
EMBED_SIZE = 256
HIDDEN_SIZE = 512
ATTENTION_DIM = 512

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))
])

# Load vocabulary and model
def load_model():
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN().eval().to(device)
    decoder = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(vocab)
    ).eval().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))

    return encoder, decoder, vocab

encoder, decoder, vocab = load_model()

def visualize_attention(image, caption_words, attention_weights):
    image = image.resize([224, 224], Image.LANCZOS)
    fig = plt.figure(figsize=(16, 8))

    rows = int(np.ceil(len(caption_words) / 5))
    cols = 5

    for i in range(len(caption_words)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(caption_words[i])
        img = np.array(image)

        attn = attention_weights[i]
        attn = attn.reshape(14, 14)
        attn = attn.detach().cpu().numpy()

        attn = Image.fromarray(attn)
        attn = attn.resize((224, 224), Image.BICUBIC)
        attn = np.array(attn)

        ax.imshow(img)
        ax.imshow(attn, cmap='gray', alpha=0.6)
        ax.axis('off')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def generate_caption_and_attention(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids, alphas = decoder.sample(features, return_alphas=True)

    sampled_ids = sampled_ids[0].cpu().numpy()
    alphas = alphas[0].cpu()

    words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            words.append(word)

    caption = ' '.join(words)
    attention_image = visualize_attention(image, words, alphas)

    return caption, attention_image

# Gradio interface
interface = gr.Interface(
    fn=generate_caption_and_attention,
    inputs=gr.Image(type="pil"),
    outputs=["text", "image"],
    title="Image Captioning with Attention",
    description="Upload an image and generate a caption with visualized attention maps."
)

if __name__ == '__main__':
    interface.launch()
