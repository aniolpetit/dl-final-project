import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]  # remove linear and pool layers
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, H, W)
        features = self.adaptive_pool(features)  # (batch_size, 2048, 14, 14)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 14, 14, 2048)
        return features

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

# Attention module
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (B, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (B, num_pixels)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha

def top_p_sampling(logits, p=0.9):
        """Performs nucleus (top-p) sampling."""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        # Create a mask for tokens to keep
        sorted_mask = cumulative_probs < p
        # Ensure at least one token is always included
        sorted_mask[..., 0] = 1

        # Zero out probabilities beyond top-p
        probs_zeroed = probs.clone().fill_(0)
        for i in range(probs.size(0)):
            idx = sorted_indices[i][sorted_mask[i]]
            probs_zeroed[i][idx] = probs[i][idx]

        # Sample from the filtered distribution
        sampled = torch.multinomial(probs_zeroed, 1).squeeze(1)
        return sampled

# Decoder with Attention
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_size, hidden_size, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        embeddings = self.embedding(captions)

        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = [l - 1 for l in lengths]
        max_len = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)

        for t in range(max_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            context, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds

        return predictions
    
    
    def sample(self, encoder_out, max_len=20, p=0.95, return_alphas=False):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        h, c = self.init_hidden_state(encoder_out)
        inputs = torch.LongTensor([1] * batch_size).to(encoder_out.device)  # <start> token id = 1

        sampled_ids = []
        alphas = [] if return_alphas else None

        for _ in range(max_len):
            embeddings = self.embedding(inputs)
            context, alpha = self.attention(encoder_out, h)  # <- Keep alpha
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([embeddings, context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            outputs = self.fc(h)

            # Apply top-p sampling
            predicted = top_p_sampling(outputs, p=p)

            sampled_ids.append(predicted.unsqueeze(1))
            inputs = predicted

            if return_alphas:
                alphas.append(alpha.unsqueeze(1))  # (B, 1, num_pixels)

        sampled_ids = torch.cat(sampled_ids, dim=1)
        
        if return_alphas:
            alphas = torch.cat(alphas, dim=1)  # (B, max_len, num_pixels)
            return sampled_ids, alphas

        return sampled_ids
