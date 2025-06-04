import os
import json
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from pycocotools.coco import COCO

import numpy as np
from PIL import Image

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def caption_length_distribution(captions, save_path):
    lengths = [len(nltk.word_tokenize(c.lower())) for c in captions]
    plt.figure(figsize=(8,5))
    plt.hist(lengths, bins=range(0, 50, 1), color='skyblue', edgecolor='black')
    plt.title('Caption Length Distribution')
    plt.xlabel('Caption Length (words)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def word_frequency(captions, save_path, top_n=30):
    tokens = []
    for c in captions:
        tokens.extend(nltk.word_tokenize(c.lower()))
    counter = Counter(tokens)
    common = counter.most_common(top_n)
    words, counts = zip(*common)
    plt.figure(figsize=(10,6))
    plt.bar(words, counts, color='lightcoral')
    plt.title(f'Top {top_n} Most Frequent Words')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(save_path)
    plt.close()

def sample_images_distribution(coco, image_dir, save_path, sample_size=1000):
    # Example: plot image size distribution for a sample
    widths, heights = [], []
    img_ids = list(coco.imgs.keys())
    for img_id in img_ids[:sample_size]:
        img_info = coco.loadImgs(img_id)[0]
        widths.append(img_info['width'])
        heights.append(img_info['height'])
    plt.figure(figsize=(8,5))
    plt.scatter(widths, heights, alpha=0.3, s=5)
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title(f'Distribution of Image Sizes (sample of {sample_size})')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    nltk.download('punkt')
    annotation_path = 'data/annotations/captions_train2014.json'
    image_dir = 'data/train2014'

    ensure_dir('exploration_results')

    coco = COCO(annotation_path)
    captions = [coco.anns[ann]['caption'] for ann in coco.anns]

    caption_length_distribution(captions, 'exploration_results/caption_length_hist.png')
    word_frequency(captions, 'exploration_results/word_freq_top30.png')
    sample_images_distribution(coco, image_dir, 'exploration_results/image_size_scatter.png')

    print("Exploration plots saved in 'exploration_results/' folder.")

if __name__ == '__main__':
    main()
