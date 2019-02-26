# -*- coding: utf-8 -*-
# 生成诗歌采用的模型

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config
from util import load_embedding, get_weight_matrix, load_vocab, load_word2idx_idx2word


class PoetryGenerator(nn.Module):
    """
    CharRNN based Poetry Generator
    """

    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size, rnn_num_layers=1, rnn_type=config.rnn_type_LSTM,
                 tie_weights=False):
        super(PoetryGenerator, self).__init__()

        # rnn网络类型
        self.rnn_type = rnn_type
        # rnn网络隐层大小
        self.rnn_hidden_size = rnn_hidden_size
        # rnn网络层数
        self.rnn_num_layers = rnn_num_layers

        # use embedding layer as first layer
        self.encoder = self.init_embedding(embedding_dim)
        self.encoder.to(config.device)

        # TODO: better CSM
        # ==============================================================================================
        # Convolutional Sentence Model (CSM) layers, compresses a line (sequence of vectors) to a vector
        # use full convolutional layers without pooling
        # ==============================================================================================
        self.csm_l1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 2, 1), stride=(1, 1, 1)),
            nn.Dropout2d()).to(config.device)
        self.csm_l2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 2, 1), stride=(1, 1, 1)),
            nn.Dropout2d()).to(config.device)
        self.csm_l3 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 1), stride=(1, 1, 1)),
            nn.Dropout2d()).to(config.device)
        self.csm_l4 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 1), stride=(1, 1, 1)),
            nn.Dropout2d()).to(config.device)

        # TODO: better context
        # ====================================================================================================
        # Context Model (CM) layers, compresses vectors of lines to one vector
        # for convenience, define 2 selectable layers for the training data is QUATRAIN which contains 4 lines
        # ====================================================================================================
        # Compress 2 lines context into 1 vector
        self.cm_21 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(2, 1), stride=(1, 1)),
            nn.Dropout2d()).to(config.device)
        # Compress 3 lines context into 1 vector
        self.cm_31 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 1), stride=(1, 1)),
            nn.Dropout2d()).to(config.device)

        # ==============================================================================================
        # Recurrent Generation Model (RGM) layers,
        # generates one word according to the previous words in the current line and the previous lines
        # ==============================================================================================
        # the inputs is concatenation of word embedding and lines vector (the same dimension as word embedding just now)
        if self.rnn_type == config.rnn_type_LSTM:
            self.rnn = nn.LSTM(input_size=self.rnn_hidden_size * 2, hidden_size=self.rnn_hidden_size,
                               num_layers=self.rnn_num_layers, batch_first=True,
                               dropout=0.5)
        elif self.rnn_type == config.rnn_type_GRU:
            self.rnn = nn.GRU(input_size=self.rnn_hidden_size * 2, hidden_size=self.rnn_hidden_size,
                              num_layers=self.rnn_num_layers, batch_first=True,
                              dropout=0.5)
        else:
            self.rnn = nn.RNN(input_size=self.rnn_hidden_size * 2, hidden_size=self.rnn_hidden_size,
                              num_layers=self.rnn_num_layers, batch_first=True,
                              dropout=0.5)
        self.rnn.to(config.device)

        self.decoder = nn.Linear(self.rnn_hidden_size, vocab_size)
        self.decoder.to(config.device)
        # tie weights, i.e. use same weight as encoder for decoder, (I learned this trick from PyTorch Example).
        if tie_weights:
            self.decoder.weight = self.encoder.weight

    def forward(self, word, sentence, hidden):

        # The code below is designed for mini-batch training
        # word embedding and reshape to (batch size, seq lenth, hidden size)
        word = word.to(config.device)
        sentence = sentence.to(config.device)

        # for 7-character quatrain, seq_length is 6
        word = self.encoder(word).view(word.size(0), -1, self.rnn_hidden_size)

        # embedding requires indices.dim() < 2, so reshape sentence to (batch size, num_lines * num_words_each_line)
        # then embed words in sentences into embedded space, and then reshape to
        # (batch size, num_channel(used in cnn), depth(num_ines), height(num_words_each_line, hidden_size)
        sentence = self.encoder(sentence.reshape(word.size(0), -1)).reshape(
            word.size(0), 1, sentence.size(-2), sentence.size(-1), self.rnn_hidden_size)

        # ===
        # CSM
        # ===
        # sentence - (batch size, channel size, depth, height, width)
        # after compressing, v - (batch size, channel size, depth, 1, width)
        v = F.leaky_relu(self.csm_l1(sentence))  # TODO: more helpful activation function
        v = F.leaky_relu(self.csm_l2(v))
        v = F.leaky_relu(self.csm_l3(v))
        if sentence.size(3) == 7:
            v = F.leaky_relu(self.csm_l4(v))
        assert v.size(3) == 1

        # ==
        # CM
        # ==
        # reshape v into 4 dimensions tensor (remove height)
        # (batch size, channel size, depth, width)
        v = v.view(word.size(0), 1, sentence.size(2), self.rnn_hidden_size)

        # 3 lines to 1 vector or 2 lines to vector
        if sentence.size(2) > 1:
            cm = self.cm_21 if sentence.size(2) == 2 else self.cm_31
            u = F.leaky_relu(cm(v))
        else:
            u = v  # if generating 2nd line, there is only 1 previous line, no need compression.

        # reshape to (batch, channel size, hidden size), remove depth
        u = u.view(word.size(0), 1, self.rnn_hidden_size)

        # for convenience again. (Forgive me - -.)
        u = u.repeat(1, word.size(1), 1)

        # ===
        # RGM
        # ===
        # The input of RGM is the concatenation of word embedding vector and sentence context vector
        uw = torch.cat([word, u], dim=2)
        # reshape to (batch size, seq length, hidden size)
        uw = uw.view(word.size(0), word.size(1), self.rnn_hidden_size * 2)
        y, hidden_out = self.rnn(uw, hidden)

        y = y.reshape(-1, self.rnn_hidden_size)
        y = self.decoder(y)
        return y, hidden_out

    def init_hidden(self, batch_size):
        # Zero initial
        # (other initialization is ok, I haven't tried others.)
        if self.rnn_type == config.rnn_type_LSTM:
            # LSTM has 2 states, one for cell state, another one for hidden state
            return (
                Variable(torch.zeros(self.rnn_num_layers * 1, batch_size, self.rnn_hidden_size, device=config.device)),
                Variable(torch.zeros(self.rnn_num_layers * 1, batch_size, self.rnn_hidden_size, device=config.device)))
        else:
            return Variable(
                torch.zeros(self.rnn_num_layers * 1, batch_size, self.rnn_hidden_size, device=config.device))

    # 初始化，并使用词嵌入方法
    def init_embedding(self, embedding_dim):
        # use `pre-trained` embedding layer, this trick is learned from machinelearningmastery.com
        _vocab = load_vocab()
        _word2idx = {w: idx for idx, w in enumerate(_vocab)}
        raw_embedding = load_embedding(config.path_embedding)
        embedding_weight = get_weight_matrix(raw_embedding, _word2idx, embedding_dim=embedding_dim)
        embedding = nn.Embedding(len(_vocab), embedding_dim=embedding_dim)
        embedding.to(config.device)
        embedding.weight = nn.Parameter(torch.from_numpy(embedding_weight).float())
        embedding.weight.to(config.device)
        return embedding


if __name__ == '__main__':
    print("TOY EXAMPLE, JUST FOR TEST!!!")

    import numpy as np
    import torch.optim as optim

    vocab = load_vocab()
    word2idx, idx2word = load_word2idx_idx2word(vocab)
    poetry = "鹤 湖 东 去 水 茫 茫	一 面 风 泾 接 魏 塘	看 取 松 江 布 帆 至	鲈 鱼 切 玉 劝 郎 尝"
    sentences = [s.split() for s in poetry.split("\t")]
    isentences = [[word2idx[w] for w in s] for s in sentences]
    print(sentences)
    print(isentences)

    batch_size = 1
    epochs = 10  # 经过长时间的训练, 程序能够"记"住一些信息

    # optimizer parameters
    # lr = 0.01
    # decay_factor = 0.00001
    # betas = (0.9, 0.999)

    pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=config.hidden_size, rnn_hidden_size=config.hidden_size,
                         tie_weights=True,
                         rnn_num_layers=config.rnn_num_layers)

    # Freeze parameters of Embedding Layer
    for param in pg.encoder.parameters():
        param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, pg.parameters())

    optimizer = optim.Adam(params=parameters, lr=config.lr, betas=config.betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / config.decay_factor)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for iss, s in enumerate(isentences[1:], start=1):
            hidden = pg.init_hidden(batch_size)
            pg.zero_grad()
            y, hidden = pg.forward(word=Variable(torch.LongTensor([s[:-1]])),
                                   sentence=Variable(torch.LongTensor([isentences[:iss]])),
                                   hidden=hidden)

            loss = criterion(y, Variable(torch.LongTensor([s[1:]])).view(-1))
            print(loss.data.numpy())
            loss.backward()
            optimizer.step()

    hidden = pg.init_hidden(batch_size)  # every epoch, we need to re-initial the hidden state
    for i, w in enumerate("松江"):
        a, hidden = pg.forward(word=Variable(torch.LongTensor([word2idx[w]])),
                               sentence=Variable(torch.LongTensor(isentences[:1])),
                               hidden=hidden)
        print("indexes", np.argmax(a.data.numpy(), axis=1))
        print([idx2word[i] for i in np.argmax(a.data.numpy(), axis=1)])
