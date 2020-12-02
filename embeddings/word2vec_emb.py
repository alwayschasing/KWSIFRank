#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
# from allennlp.commands.elmo import ElmoEmbedder
from .elmoformanylangs import Embedder
import numpy as np
import time
import torch
def load_word2vec_emb(input_file):
    fp = open(input_file, "r", encoding="utf-8")
    num = 0; size = 0
    vocab_emb = {}
    for idx, line in enumerate(fp):
        if idx == 0:
            parts = line.rstrip('\n').split(' ')
            num = int(parts[0])
            size = int(parts[1])
            continue
        parts = line.rstrip('\n').split(' ')
        word = parts[0].lower()
        vec = [float(x) for x in parts[1:]]
        vocab_emb[word] = vec

    # logger.info("num:%d, vocab_emb size:%d" %(num, len(vocab_emb)))
    # assert num == len(vocab_emb)
    return vocab_emb

class Word2VecEmbeddings():
    """
        ELMo
        https://allennlp.org/elmo

    """
    def __init__(self, w2v_embedding_file):
        self.word_embedding = load_word2vec_emb(w2v_embedding_file)
        self.embedding_size = 100

    def get_tokenized_words_embeddings(self, sents_tokened):
        elmo_embedding = []
        for sent in sents_tokened:
            sent_embs = []
            for w in sent:
                sent_embs.append(np.array(self.word_embedding[w]))
            elmo_embedding.append(sent_embs)

        # elmo_embedding = [np.pad(emb, pad_width=((0,0),(0,max_len-emb.shape[1]),(0,0)) , mode='constant') for emb in elmo_embedding]
        elmo_embedding = np.asarray(elmo_embedding)
        return elmo_embedding

# if __name__ == '__main__':
#     sents = [['今', '天', '天气', '真', '好', '啊'],
#              ['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
#     elmo = WordEmbeddings()
#     embs = elmo.get_tokenized_words_embeddings(sents)
#     print("OK")
