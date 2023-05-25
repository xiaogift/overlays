#!/usr/bin/env python
# ============================================================================== #
# Image overlay test
# Powered by xiaolis@outlook.com 202305
# ============================================================================== #
import random, torch
from torchvision.datasets import CIFAR10, STL10
import matplotlib.pyplot as plt
from PIL import Image

DATADIR = '../data'

def get_10_images(dataset, order=True):
    result = []
    for i in range(100):
        img, lab = dataset[i][0],dataset[i][1]
        if order and (lab!=i):
            for j in range(1000):
                lab = dataset[j+10][1]
                if (lab==i): 
                    img = dataset[j+10][0]
                    break
            result.append([img, lab])
        else: result.append([img, lab])
    return result

def plot_samples(samples, img_size):
    result_image = Image.new('RGB', (img_size*10,img_size*2), color=(255, 255, 255))
    for i in range(10):
        current_image = samples[i][0]
        result_image.paste(current_image, (i*img_size, img_size))
    plt.imshow(result_image)
    plt.axis('off'); plt.show()
    print([samples[i][1] for i in range(10)])
    return result_image

def xsamples(dataset, size):
    dataset = list(dataset)
    random.shuffle(dataset)
    a = plot_samples(get_10_images(dataset,order=True), size)
    random.shuffle(dataset)
    b = plot_samples(get_10_images(dataset,order=True), size)
    random.shuffle(dataset)
    c = plot_samples(get_10_images(dataset,order=False), size)
    plt.imshow(Image.blend(a,b,0.5))
    plt.axis('off'); plt.show()
    plt.imshow(Image.blend(a,c,0.5))
    plt.axis('off'); plt.show()

# ============================================================================== #
if __name__ == '__main__': 
    xsamples(CIFAR10( root=DATADIR, train=True, download=True),32)
    xsamples(STL10(root=DATADIR, split='train', download=True),96)
