import os, torch
dataset_dir = "flickr8k"
image_dir = os.path.join(dataset_dir, "Images")
captions_file = os.path.join(dataset_dir, "captions.txt")
image_path = image_dir
captions_path = dataset_dir
class CFG:
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3  
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # 이미지 인코더, 텍스트 인코더 둘 다
    trainable = True # 이미지 인코더, 텍스트 인코더 둘 다
    temperature = 1.0

    # 이미지사이즈
    size = 224

    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1