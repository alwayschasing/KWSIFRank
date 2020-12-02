#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
# from allennlp.commands.elmo import ElmoEmbedder
from .elmoformanylangs import Embedder
import numpy as np
import time
import torch
class WordEmbeddings():
    """
        ELMo
        https://allennlp.org/elmo

    """

    def __init__(self,
                 model_path=r'../auxiliary_data/zhs.model/', cuda_device=2, batch_size=8):
        self.cuda_device=cuda_device
        self.elmo = Embedder(model_path, batch_size=batch_size, cuda_device=cuda_device)

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param tokenized_sents: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), )dimension of embeddings
        """
        max_len = max([len(sent) for sent in sents_tokened])
        st = time.time()
        elmo_embedding = self.elmo.sents2elmo(sents_tokened,output_layer=-2)
        ed = time.time()
        cost = int((ed - st)*1000)
        # print("[cuda_cost] %dms"%(cost))
        elmo_embedding = [np.pad(emb, pad_width=((0,0),(0,max_len-emb.shape[1]),(0,0)) , mode='constant') for emb in elmo_embedding]
        # elmo_embedding = torch.from_numpy(np.array(elmo_embedding))
        elmo_embedding = np.array(elmo_embedding)
        return elmo_embedding

# if __name__ == '__main__':
#     sents = [['今', '天', '天气', '真', '好', '啊'],
#              ['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
#     elmo = WordEmbeddings()
#     embs = elmo.get_tokenized_words_embeddings(sents)
#     print("OK")
