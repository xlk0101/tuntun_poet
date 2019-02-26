# -*- coding: utf-8 -*-

import os
import torch

# device
IS_USE_CUDA = torch.cuda.is_available()
# IS_USE_CUDA = False
device = torch.device("cuda" if IS_USE_CUDA else "cpu")
# IS_DEBUG
IS_DEBUG = True
SAVE_CHECKPOINT = True

# training hyper-parameters
mode = "train_1000"  # used for save checkpoint file
epochs = 1000
batch_size = 1024
embedding_dim = 256
hidden_size = 256
rnn_num_layers = 3
lr = 0.01
decay_factor = 1.004
betas = (0.9, 0.999)

# dir, for convenience
dir_chkpt = os.path.join("checkpoints")
dir_data = os.path.join(os.path.dirname(__file__), os.pardir, "data")
dir_rnnpg = os.path.join(dir_data, "rnnpg_data_emnlp-2014")
dir_poemlm = os.path.join(dir_rnnpg, "partitions_in_Table_2", "poemlm")

# file path
path_pingshuiyun = os.path.join(dir_data, "pingshuiyun.txt")  # Pingshuiyun contains the Ping/Ze of words, incomplete
path_shixuehanying = os.path.join(dir_data, "shixuehanying.txt")  # Shixuehanying contains the category of words
path_embedding = os.path.join(dir_data, "embedding_word2vec.txt")
path_vocab = os.path.join(dir_data, "vocab.txt")

# rnn type
rnn_type_LSTM = "LSTM"
rnn_type_GRU = "GRU"

# poem length
len_7 = 7
len_5 = 5

# poem mode
mode_normal = 0  # 普通诗
mode_acrostic = 1  # 藏头诗

# The most common tonals for quatrain
# (0 for either, 1 for Ping, -1 for Ze)
QUATRAIN_5 = [
    [[0, -1, 1, 1, -1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],  # 首句仄起仄收
    [[0, -1, -1, 1, 1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],  # 首句仄起平收
    [[0, 1, 1, -1, -1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]],  # 首句平起仄收
    [[1, 1, -1, -1, 1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]]  # 首句平起平收
]

QUATRAIN_7 = [
    [[0, 1, 0, -1, -1, 1, 1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],  # 首句平起平收
    [[0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],  # 首句平起仄收
    [[0, -1, 1, 1, -1, -1, 1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]],  # 首句仄起平收
    [[0, -1, 0, 1, 1, -1, -1], [0, 0, -1, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]]  # 首句仄起仄收
]
