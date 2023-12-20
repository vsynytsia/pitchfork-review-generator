import torch


class UrlConfig:
    PITCHFORK = "https://pitchfork.com"
    ALBUM_REVIEWS = f"{PITCHFORK}/reviews/albums"


class DataFoldersConfig:
    RAW = "data/raw"
    PREPROCESSED = "data/preprocessed"


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_EPOCHS = 100
    BATCH_SIZE = 32
    SEQ_LEN = 150
    N_HIDDEN = 512
    N_LAYERS = 4
    DROPOUT_PROB = 0.5
