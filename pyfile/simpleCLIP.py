import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import zipfile

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import subprocess

print("Downloading dataset from Kaggle...")
subprocess.run(["kaggle", "datasets", "download", "-d", "adityajn105/flickr8k"])

print("Unzipping dataset...")
with zipfile.ZipFile("flickr8k.zip", 'r') as zip_ref:
    zip_ref.extractall("flickr8k")

dataset_dir = "flickr8k"
image_dir = os.path.join(dataset_dir, "Images")
captions_file = os.path.join(dataset_dir, "captions.txt")

dataset = "8k"

if dataset == "8k":
    df = pd.read_csv(captions_file)
    df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
    df.to_csv(os.path.join(dataset_dir, "captions.csv"), index=False)

    df = pd.read_csv(os.path.join(dataset_dir, "captions.csv"))
    image_path = image_dir
    captions_path = dataset_dir


df.head()

import AvgMeter, build_loaders, CFG, CLIP, cross_entropy, Encoder, Epoch, get_TFs, main, make_train_valid_dfs, ProjectionHead

# A simple Example

batch_size = 4
dim = 256
embeddings = torch.randn(batch_size, dim)
out = embeddings @ embeddings.T
print(F.softmax(out, dim=-1))

main()

_, valid_df = make_train_valid_dfs()
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")

find_matches(model,
             image_embeddings,
             query="dogs on the grass",
             image_filenames=valid_df['image'].values,
             n=9)