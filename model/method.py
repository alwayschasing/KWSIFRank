#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from model import input_representation
import torch
import time
import pickle
import numpy
import logging
from model import textrank, util
logger = util.get_logger(__name__, debug=1)

wnl=nltk.WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def cos_sim_gpu(x,y):
    assert x.shape[0]==y.shape[0]
    zero_tensor = torch.zeros((1, x.shape[0])).cuda()
    # zero_list = [0] * len(x)
    if x == zero_tensor or y == zero_tensor:
        return float(1) if x == y else float(0)
    xx, yy, xy = 0.0, 0.0, 0.0
    for i in range(x.shape[0]):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i]
    return 1.0 - xy / np.sqrt(xx * yy)

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

def cos_sim_transformer(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    a = vector_a.detach().numpy()
    b = vector_b.detach().numpy()
    a=np.mat(a)
    b=np.mat(b)

    num = float(a * b.T)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

def get_dist_cosine(emb1, emb2, sent_emb_method="elmo",elmo_layers_weight=[0.0,1.0,0.0]):
    sum = 0.0
    assert emb1.shape == emb2.shape
    if(sent_emb_method=="elmo"):

        for i in range(0, 3):
            a = emb1[i]
            b = emb2[i]
            sum += cos_sim(a, b) * elmo_layers_weight[i]
        return sum

    elif(sent_emb_method=="elmo_transformer"):
        sum = cos_sim_transformer(emb1, emb2)
        return sum

    elif(sent_emb_method=="doc2vec"):
        sum=cos_sim(emb1,emb2)
        return sum

    elif (sent_emb_method == "glove"):
        sum = cos_sim(emb1, emb2)
        return sum
    return sum

def get_all_dist(candidate_embeddings_list, text_obj, dist_list):
    '''
    :param candidate_embeddings_list:
    :param text_obj:
    :param dist_list:
    :return: dist_all
    '''

    dist_all={}
    for i, emb in enumerate(candidate_embeddings_list):
        phrase = text_obj.keyphrase_candidate[i][0]
        phrase = phrase.lower()
        phrase = wnl.lemmatize(phrase)
        if(phrase in dist_all):
            #store the No. and distance
            dist_all[phrase].append(dist_list[i])
        else:
            dist_all[phrase]=[]
            dist_all[phrase].append(dist_list[i])
    return dist_all

def get_final_dist(dist_all, method="average"):
    '''
    :param dist_all:
    :param method: "average"
    :return:
    '''

    final_dist={}

    if(method=="average"):

        for phrase, dist_list in dist_all.items():
            sum_dist = 0.0
            for dist in dist_list:
                sum_dist += dist
            if (phrase in stop_words):
                sum_dist = 0.0
            final_dist[phrase] = sum_dist/float(len(dist_list))
        return final_dist

def softmax(x):
    # x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_position_score(keyphrase_candidate_list, position_bias):
    length = len(keyphrase_candidate_list)
    position_score ={}
    for i,kc in enumerate(keyphrase_candidate_list):
        np = kc[0]
        p = kc[1][0]
        np = np.lower()
        np = wnl.lemmatize(np)
        if np in position_score:

            position_score[np] += 0.0
        else:
            position_score[np] = 1/(float(i)+1+position_bias)
    score_list=[]
    for np,score in position_score.items():
        score_list.append(score)
    score_list = softmax(score_list)

    i=0
    for np, score in position_score.items():
        position_score[np] = score_list[i]
        i+=1
    return position_score

def SIFRank(text, SIF, en_model, method="average", N=15,
            sent_emb_method="elmo", elmo_layers_weight=[0.0, 1.0, 0.0], if_DS=True, if_EA=True, kwdict=None, kw_info=None, cut_dict=False, seg_only=False):
    """
    :param text_obj:
    :param sent_embeddings:
    :param candidate_embeddings_list:
    :param sents_weight_list:
    :param method:
    :param N: the top-N number of keyphrases
    :param sent_emb_method: 'elmo', 'glove'
    :param elmo_layers_weight: the weights of different layers of ELMo
    :param if_DS: if take document segmentation(DS)
    :param if_EA: if take  embeddings alignment(EA)
    :return:
    """
    text_obj = input_representation.InputTextObj(en_model, text, kw_dict=kwdict, cut_dict=cut_dict, seg_only=seg_only)
    sent_embeddings, candidate_embeddings_list = SIF.get_tokenized_sent_embeddings(text_obj,if_DS=if_DS,if_EA=if_EA)
    dist_list = []
    for i, emb in enumerate(candidate_embeddings_list):
        dist = get_dist_cosine(sent_embeddings, emb, sent_emb_method, elmo_layers_weight=elmo_layers_weight)
        dist_list.append(dist)
    dist_all = get_all_dist(candidate_embeddings_list, text_obj, dist_list)
    dist_final = get_final_dist(dist_all, method='average')
    dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)

    phrase_freq = dict()
    max_freq,min_freq = 1,1
    for phrase, dist_list in dist_all.items():
        freq = len(dist_list)
        phrase_freq[phrase] = freq
        if freq < min_freq:
            min_freq = freq
        if freq > max_freq:
            max_freq = freq

    return dist_sorted[0:N]

def SIFRank_plus(text, SIF, en_model, method="average", N=15,
            sent_emb_method="elmo", elmo_layers_weight=[1.0, 0.0, 0.0], if_DS=True, if_EA=True, position_bias = 3.4, kwdict=None, kw_info=None, cut_dict=False, seg_only=False, use_pos=False):
    """
    :param text_obj:
    :param sent_embeddings:
    :param candidate_embeddings_list:
    :param sents_weight_list:
    :param method:
    :param N: the top-N number of keyphrases
    :param sent_emb_method: 'elmo', 'glove'
    :param elmo_layers_weight: the weights of different layers of ELMo
    :return:
    """
    text_obj = input_representation.InputTextObj(en_model, text, kw_dict=kwdict, cut_dict=cut_dict, seg_only=seg_only, use_pos=use_pos)
    st = time.time()
    sent_embeddings, candidate_embeddings_list = SIF.get_tokenized_sent_embeddings(text_obj,if_DS=if_DS,if_EA=if_EA)
    #logging.debug("sent_embeddings:%s\ncandidate_embeddings_list:%s" %(sent_embeddings, candidate_embeddings_list))
    ed = time.time()
    cost = int((ed - st)*1000)
    #logging.debug("[emb_cost] %dms" %(cost))
    position_score = get_position_score(text_obj.keyphrase_candidate, position_bias)
    if len(position_score) == 0:
        average_score = 0
    else:
        average_score = sum(position_score.values())/(float)(len(position_score))

    dist_list = []
    for i, emb in enumerate(candidate_embeddings_list):
        dist = get_dist_cosine(sent_embeddings, emb, sent_emb_method, elmo_layers_weight=elmo_layers_weight)
        #logging.debug("%s, cosine:%f" % (text_obj.keyphrase_candidate[i], dist))
        dist_list.append(dist)
    dist_all = get_all_dist(candidate_embeddings_list, text_obj, dist_list)
    dist_final = get_final_dist(dist_all, method='average')

    # for np,dist in dist_final.items():
    #     if np in position_score:
    #         dist_final[np] = dist*position_score[np]/average_score

    logger.info("[check sif] %s" % (dist_final))
    
    min_s = 1.0; max_s = 0.0
    for w,s in dist_final.items():
        if s < min_s:
            min_s = s
        if s > max_s:
            max_s = s

    # dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
    word_textrank = textrank.get_textrank(text_obj.sentence_words, window=10)
    new_textrank = {}
    for k,v in word_textrank.items():
        if k not in kwdict:
            continue
        new_textrank[k] = v

    min_sr = 1.0; max_sr = 0.0; denominate_sr = 1.0
    for w,s in dist_final.items():
        if s < min_sr:
            min_sr = s
        if s > max_sr:
            max_sr = s
    denominate_sr = max_sr - min_sr
    
    min_tr = 1.0; max_tr = 0.0
    for w,s in new_textrank.items():
        if s < min_tr:
            min_tr = s
        if s > max_tr:
            max_tr = s
    denominate_tr = max_tr - min_tr

    final_items = []
    if len(word_textrank) > 0:
        for w,s in dist_final.items():
            if w not in new_textrank:
                continue
            norm_sr = (s - min_sr) / denominate_sr 
            norm_tr = (new_textrank[w] - min_tr) / denominate_tr 
            final_items.append((w, norm_sr*norm_tr, norm_sr, norm_tr))
    final_items = sorted(final_items, key=lambda x: x[1], reverse=True)
    logger.info("[check method]dist_final:%s\nword_textrank:%s\nfinal_items:%s" % (dist_final, word_textrank, final_items))
    
    return final_items


