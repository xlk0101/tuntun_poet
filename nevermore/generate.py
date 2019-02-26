# -*- coding: utf-8 -*-

import copy
import os
import random
from queue import Queue

import torch
from torch.autograd import Variable
import kenlm

import config
from dataset import Poemsets
from model import PoetryGenerator
import util

TAG = "generate"

# 加载词汇表，从训练集中提取出来的数据
vocab = util.load_vocab()
# 提取前2500个频率最高的词
commons = vocab[:2500]  # most frequent words
word2idx, idx2word = util.load_word2idx_idx2word(vocab)
# 高频词对应的index
icommons = [word2idx[w] for w in commons]
# 初始化平仄和韵脚，用于生成诗句时约束使用
pingzes, yuns = util.read_pingshuiyun(use_index=True)
# 初始化word2vec网络，从qtotal中训练而来
word2vec = util.load_word2vec()
# 加载kenlm模型
LM = os.path.join(os.path.dirname(__file__), '..', 'data', 'rnnpg_data_emnlp-2014',
                  'partitions_in_Table_2', 'poemlm', 'qts.klm')
kmodel = kenlm.LanguageModel(LM)

# poemset = Poemsets(os.path.join(config.dir_rnnpg, "partitions_in_Table_2", "rnnpg", "qtrain_7"))
pg = PoetryGenerator(vocab_size=len(vocab), embedding_dim=config.embedding_dim, rnn_hidden_size=config.hidden_size,
                     rnn_num_layers=config.rnn_num_layers,
                     tie_weights=True)
pg.to(config.device)
loadfile = os.path.join(config.dir_chkpt, config.mode, "{}_new2.chkpt".format(config.epochs - 1))
checkpoint = torch.load(loadfile)
util.loginfo(TAG, "load checkpoint from:{}".format(loadfile))
pg.load_state_dict(checkpoint)


def get_yun(w):
    for head, words in yuns.items():
        if w in words:
            return head
    raise ValueError("Can not find Yun for {w}".format(w=w))


def giveme_a_poetry(firstline, pattern, generator=pg, firstwords=None, word2vec=word2vec, n_candicates=2):
    """
    :param firstline:
    :param generator:
    :param word2vec:
    :param pattern:
    :return:
    """
    lines = [firstline]  # lines represent the poem
    yun_head = get_yun(firstline[-1])
    generated_firstwords = []

    for l in range(len(pattern) - len(lines)):
        # ========================================
        # choose 1st word for each following line
        # ========================================
        if firstwords and l < len(firstwords):
            current_line = [firstwords[l]]
        else:
            # to be honest, I have no idea how to choose the 1st word,
            # so just find the most similar word to previous one
            # for word2vec use str, here we have to convert the first word in last line from index to word
            word = idx2word[lines[-1][0]]
            generated_firstwords.append(word)
            most_similar_words = [word2idx[p[0][0]] for p in word2vec.wv.most_similar(positive=word, topn=250)]
            most_similar_words = [  # remove those dissatisfy Ping/Ze.
                w for w in most_similar_words
                if w in icommons and w in pingzes.keys() and (
                        pattern[l + 1][0] == 0 or (pattern[l + 1][0] == -1) ^ pingzes[w])
            ]
            # current_line = [random.sample(most_similar_words[:5], 1)[0]]  #
            ken_scores = [kmodel.score(" ".join(generated_firstwords) + " " + w) for w in
                          (idx2word[i] for i in most_similar_words)]
            rd_ken_scores = random.sample(ken_scores, 3)  # random drop
            chosen = ken_scores.index(max(ken_scores))
            current_line = [chosen]

        # TODO: Antithesis (前后对仗)
        # temporarily cancel the feature by setting probability < 0.0
        if len(lines) in (1, 3) and (len(set(lines[-1])) < len(lines[-1])) and random.random() < 0.0:
            previous_line = [idx2word[w] for w in lines[l]]
            for iw, w in enumerate(previous_line[1:], start=1):  # start from the second word
                word_candicates = word2vec.wv.most_similar(
                    positive=[previous_line[0], w],
                    negative=[idx2word[current_line[0]]], topn=500, restrict_vocab=2000)
                word_candicates = [wc[0] for wc in word_candicates]
                word_candicates = [
                    w for w in word_candicates
                    if
                    word2idx[w] in pingzes.keys() and (pattern[l + 1][iw] == 0 or pattern[l + 1][iw] == -1) ^ pingzes[
                        word2idx[w]]
                ]
                current_line.append(word2idx[word_candicates[0]])
            lines.append(current_line)
        else:
            candicates = Queue()
            candicate_scores = Queue()
            hiddens = Queue()

            # ===========
            # beam search
            # ===========
            # init candicates queue and hidden states queue
            candicates.put(copy.deepcopy(current_line))
            candicate_scores.put([1])
            hiddens.put(generator.init_hidden(batch_size=1))
            # import pdb
            # pdb.set_trace()
            for e in range(len(pattern[0]) - 1):
                for n in range(n_candicates ** e):
                    cdct = candicates.get()
                    score = candicate_scores.get()
                    hidden = hiddens.get()
                    # 用训练好的RNN预测下一个字
                    generated, hidden = generator.forward(word=Variable(torch.LongTensor([cdct[-1]])),
                                                          sentence=Variable(torch.LongTensor(lines)),
                                                          hidden=hidden)
                    if e == len(pattern[0]) - 2 and len(lines) in (1, 3):
                        topv, topi = generated.data.topk(3500)
                        # remove those dissatisfy Ping/Ze
                        # i.item() in yuns[yun_head] and
                        topi = [int(i) for i in topi.cpu().numpy()[0]
                                if i in pingzes.keys() and
                                (pattern[l + 1][len(cdct)] == 0 or (pattern[l + 1][len(cdct)] == -1) ^ pingzes[i])]
                    else:
                        topv, topi = generated.data.topk(2500)
                        topi = [int(i) for i in topi.cpu().numpy()[0]
                                # remove those dissatisfy Ping/Ze
                                if i in pingzes.keys() and
                                (pattern[l + 1][len(cdct)] == 0 or (pattern[l + 1][len(cdct)] == -1) ^ pingzes[i])]
                    try:
                        for m in range(n_candicates):
                            if m < len(topi):
                                lth_index = topi[m]
                                copied_cdct = copy.deepcopy(cdct)
                                copied_cdct.append(lth_index)
                                copied_score = copy.deepcopy(score)
                                copied_score.append(generated.data.cpu().numpy()[0][lth_index])
                                candicates.put(copied_cdct)
                                candicate_scores.put(copied_score)
                                hiddens.put(hidden)
                    except Exception as ecpt:
                        raise ecpt
            candicates = [candicates.get() for q in range(candicates.qsize())]
            candicate_scores = [candicate_scores.get() for q in range(candicate_scores.qsize())]

            word_candicates = [[idx2word[idx] for idx in candicate] for candicate in candicates]
            if len(word_candicates) > 0:
                candicate_scores = [sum(cs) for cs in candicate_scores]
                min_score = min(candicate_scores)
                score_range = max(candicate_scores) - min(candicate_scores)
                normalized_scores = [(cs - min_score) / score_range for cs in candicate_scores]
                cohesion_scores = [kmodel.score(" ".join(words)) for words in word_candicates]
                cohesion_min_score = min(cohesion_scores)
                cohesion_score_range = max(cohesion_scores) - cohesion_min_score
                normalized_cohesion_scores = [(cs - cohesion_min_score) / cohesion_score_range for cs in
                                              cohesion_scores]
                scores = [a * 0.15 + b * 0.85 for a, b in zip(normalized_scores, normalized_cohesion_scores)]
                max_score = max(scores)
                max_index = scores.index(max_score)
                chosen = candicates[max_index]

                # TODO: more candicate lines, and evaluate interline cohesion
                # current_line = [word2idx[w] for w in cohesion_scores[0][0]]  # choose the best one
                current_line = chosen
                lines.append(current_line)
            else:
                util.loginfo(TAG, "the word_candicates is null, the candicates is {}".format(len(candicates)))
    return lines


if __name__ == '__main__':
    import argparse
    from firstline import make_firstline

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--qtype', type=int, default=config.len_7,
                           help="Specify 5-characters quatrain or 7-characters quatrain.")
    argparser.add_argument('-w', '--qtopic', type=str, default="春",
                           help="Specify topic for quatrain.")
    argparser.add_argument('-m', '--qmode', type=int, default=0,
                           help="Specify mode for quatrain,the default is nomal.")
    args = argparser.parse_args()

    if args.qtype == config.len_5 or args.qtype == config.len_7:
        mode = args.qmode
        firstword = []
        string = args.qtopic
        for i in range(len(string)):
            if i > 0:
                firstword.append(word2idx[string[i]])
        # first line, its mode, pattern, mutual information
        fline, _, pattern, mutual_info = make_firstline(str(args.qtype) + " " + args.qtopic, mode=mode)
        ifline = [word2idx[w] for w in fline]  # represent as list of indexes
        if mode == config.mode_acrostic:
            poem = giveme_a_poetry(ifline, firstwords=firstword, pattern=pattern)
        else:
            poem = giveme_a_poetry(ifline, pattern=pattern)
        print("\r\n".join(["".join([idx2word[c] for c in l]) for l in poem]))
    else:
        print("the number of the poem only can be 5 or 7!")
