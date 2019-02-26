# -*- coding: utf-8 -*-
# 模型训练方法

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Poemsets
from model import PoetryGenerator
import config
from util import load_word2idx_idx2word, load_vocab, secformat

# 加载词汇表，从训练集中提取出来的数据
vocab = load_vocab()
word2idx, idx2word = load_word2idx_idx2word(vocab)

# 加载诗集训练数据，加载全部训练数据，包含5言和7言
poemset = Poemsets(os.path.join(config.dir_rnnpg, "partitions_in_Table_2", "rnnpg", "qtrain_7"))
poemloader = DataLoader(poemset, batch_size=config.batch_size, shuffle=True)

# note embedding size is the same as rnn_hidden_size at this version
pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=config.embedding_dim, rnn_hidden_size=config.hidden_size,
                     rnn_num_layers=config.rnn_num_layers,
                     tie_weights=True)
pg.to(config.device)

# Freeze parameters of Embedding Layer
for params in pg.encoder.parameters():
    params.requires_grad = False

parameters = filter(lambda p: p.requires_grad, pg.parameters())

# Adam 优化方法
optimizer = optim.Adam(params=parameters, lr=config.lr, betas=config.betas, weight_decay=1e-7)
# SGD 优化方法
# optimizer = optim.SGD(params=parameters, lr=config.lr,momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / config.decay_factor)
# loss function
criterion = nn.CrossEntropyLoss()
# test
# criterion = nn.NLLLoss()

# ====================================================================
# uncomment following 5 lines to load checkpoint file
# and adjust learning rate, it's useful when training with checkpoint file
# ====================================================================
# checkpoint = torch.load(os.path.join(config.dir_chkpt, mode, "288_new.chkpt"))
# pg.load_state_dict(checkpoint)
# for i in range(289):
#     scheduler.step()
# for epoch in range(288, epochs):  # uncomment this line for avoiding overwriting exsiting checkpoint file
# 模型训练过程
starttime = time.clock()
for epoch in range(config.epochs):
    scheduler.step()  # update learning rate

    losses = []
    for poems in poemloader:

        # train with line 2
        hidden = pg.init_hidden(len(poems))
        pg.zero_grad()
        y, hidden = pg.forward(word=Variable(torch.LongTensor(poems[:, 1, :-1])),
                               sentence=Variable(torch.LongTensor(poems[:, :1])),
                               hidden=hidden)
        loss = criterion(y, Variable(torch.LongTensor(poems[:, 1, 1:])).reshape(-1).to(config.device))
        loss.backward()
        optimizer.step()

        # train with line 3
        hidden = pg.init_hidden(len(poems))
        pg.zero_grad()
        y, hidden = pg.forward(word=Variable(torch.LongTensor(poems[:, 2, :-1])),
                               sentence=Variable(torch.LongTensor(poems[:, :2])),
                               hidden=hidden)
        loss = criterion(y, Variable(torch.LongTensor(poems[:, 2, 1:])).reshape(-1).to(config.device))
        loss.backward()
        optimizer.step()

        # train with line 4
        hidden = pg.init_hidden(len(poems))
        pg.zero_grad()
        y, hidden = pg.forward(word=Variable(torch.LongTensor(poems[:, 3, :-1])),
                               sentence=Variable(torch.LongTensor(poems[:, :3])),
                               hidden=hidden)
        loss = criterion(y, Variable(torch.LongTensor(poems[:, 3, 1:])).reshape(-1).to(config.device))
        loss.backward()
        optimizer.step()

        print("epoch-{} loss-{}".format(epoch, loss.item()))
        # the following lines is used for checking whether the network works well.
        # hidden = pg.init_hidden(1)
        # _, hidden = pg.forward(w=Variable(torch.LongTensor([[word2idx[w] for w in list("一")]])),
        #                        s=Variable(torch.LongTensor([[word2idx[w] for w in list("两个黄鹂鸣翠柳")]])),
        #                        hidden=hidden)
        # _, hidden = pg.forward(w=Variable(torch.LongTensor([[word2idx[w] for w in list("行")]])),
        #                        s=Variable(torch.LongTensor([[word2idx[w] for w in list("两个黄鹂鸣翠柳")]])),
        #                        hidden=hidden)
        # a, hidden = pg.forward(w=Variable(torch.LongTensor([[word2idx[w] for w in list("白")]])),
        #                        s=Variable(torch.LongTensor([[word2idx[w] for w in list("两个黄鹂鸣翠柳")]])),
        #                        hidden=hidden)
        # print("indexes", np.argmax(a.data.numpy(), axis=1))
        # print([idx2word[i] for i in np.argmax(a.data.numpy(), axis=1)])
    save_dir = "{dir}/{mode}".format(dir=config.dir_chkpt, mode=config.mode)
    save_name = "{dir}/{mode}/{epoch}_new2.chkpt".format(dir=config.dir_chkpt, mode=config.mode, epoch=epoch)
    if config.SAVE_CHECKPOINT and epoch == config.epochs - 1:
        if os.path.exists(save_dir) is not True:
            os.makedirs(save_dir)
        torch.save(pg.state_dict(), save_name)

print("运行结束，花费时间：", secformat(time.clock() - starttime))
