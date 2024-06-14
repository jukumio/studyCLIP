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

os.environ['KAGGLE_USERNAME'] = "juuhoney"
os.environ['KAGGLE_KEY'] = "11898867"

print("Downloading dataset from Kaggle...")
# Kaggle 데이터셋 다운로드 명령어 실행
subprocess.run(["kaggle", "datasets", "download", "-d", "adityajn105/flickr8k"])

print("Unzipping dataset...")
with zipfile.ZipFile("flickr8k.zip", 'r') as zip_ref:
    zip_ref.extractall("flickr8k")

# 압축 해제된 데이터 경로 설정
dataset_dir = "flickr8k"
image_dir = os.path.join(dataset_dir, "Images")
captions_file_8k = os.path.join(dataset_dir, "captions.txt")

dataset = "8k"  # "8k" 또는 "30k"로 설정

if dataset == "8k":
    # 캡션 파일 경로
    captions_file = captions_file_8k

    # 데이터프레임 로드 및 처리
    df = pd.read_csv(captions_file)
    df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
    df.to_csv(os.path.join(dataset_dir, "captions.csv"), index=False)

    # 데이터프레임 다시 로드
    df = pd.read_csv(os.path.join(dataset_dir, "captions.csv"))
    image_path = image_dir
    captions_path = dataset_dir

'''elif dataset == "30k":
  df = pd.read_csv("/content/flickr30k_images/results.csv", delimiter="|")
  df.columns = ['image', 'caption_number', 'caption']
  df['caption'] = df['caption'].str.lstrip()
  df['caption_number'] = df['caption_number'].str.lstrip()
  df.loc[19999, 'caption_number'] = "4"
  df.loc[19999, 'caption'] = "A dog runs across the grass ."
  ids = [id_ for id_ in range(len(df) // 5) for _ in range(5)]
  df['id'] = ids
  df.to_csv("captions.csv", index=False)
  image_path = "/content/flickr30k_images/flickr30k_images"
  captions_path = "/content"'''

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