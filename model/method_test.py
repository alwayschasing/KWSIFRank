#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

import numpy as np
import math
import logging
import nltk
from nltk.corpus import stopwords
from model import input_representation
import torch
import time
import pickle
import numpy
import logging

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
            sent_emb_method="elmo", elmo_layers_weight=[0.0, 1.0, 0.0], if_DS=True, if_EA=True, kwdict=None, kw_info=None, cut_dict=False, seg_only=False, logger=logging.getLogger()):
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
    text_obj = input_representation.InputTextObj(en_model, text, kw_dict=kwdict, cut_dict=cut_dict, seg_only=seg_only, logger=logger)
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

    top_dist = dist_sorted[0:N]
    top_kw_map = dict()
    for kw,score in top_dist:
        top_kw_map[kw] = [score]

    for idx, (np,dist) in enumerate(top_dist):
        freq_weight = (phrase_freq[np] - min_freq + 1)/float(max_freq - min_freq + 1)
        #print("%s %f %f, freq:%d, max:%d, min:%d" % (np, dist, freq_weight, phrase_freq[np], max_freq, min_freq))
        if kw_info is not None:
            if np in kw_info:
                idf = kw_info[np][0]
                if idf > 0.0 and idf < 4.0:
                    idf_weight = 0.5
        top_kw_map[np].extend([freq_weight, idf_weight])

        top_dist[idx] = (np, dist*freq_weight)
    top_sorted = sorted(top_dist, key=lambda x:x[1], reverse=True)
    final_kw_list = []
    for kw,score in top_sorted:
        final_kw_list.append((kw, top_kw_map[kw][0]))
    return final_kw_list

def SIFRank_plus(text, SIF, en_model, method="average", N=15,
            sent_emb_method="elmo", elmo_layers_weight=[1.0, 0.0, 0.0], if_DS=True, if_EA=True, position_bias = 3.4,
            kwdict=None, kw_info=None, cut_dict=False, seg_only=False, use_pos=False, logger=logging.getLogger(), check=False):
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
    text_obj = input_representation.InputTextObj(en_model, text, kw_dict=kwdict, cut_dict=cut_dict, seg_only=seg_only, use_pos=use_pos, logger=logger)
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
    phrase_freq = dict()
    max_freq,min_freq = 1,1
    for phrase, dist_list in dist_all.items():
        freq = len(dist_list)
        phrase_freq[phrase] = freq
        if freq < min_freq:
            min_freq = freq
        if freq > max_freq:
            max_freq = freq

    dist_final = get_final_dist(dist_all, method='average')

    for np,dist in dist_final.items():
        if np in position_score:
            dist_final[np] = dist*position_score[np]/average_score

    dist_sorted = sorted(dist_final.items(), key=lambda x: x[1], reverse=True)
    check_kw_score = " ".join(["%s:%s" % (k,s) for k,s in dist_sorted])
    logger.info("[check_kw_score]%s" %(check_kw_score))
    top_dist = dist_sorted[0:N]
    top_kw_map = dict()
    for kw,score in top_dist:
        top_kw_map[kw] = [score]

    for idx, (np,dist) in enumerate(top_dist):
        freq_weight = (phrase_freq[np] - min_freq + 1)/float(max_freq - min_freq + 1)
        #if phrase_freq[np] < 2:
        #    freq_weight = freq_weight*0.5
        idf_weight = 1.0
        if kw_info is not None:
            if np in kw_info:
                idf = kw_info[np][0]
                if idf >= 13.0:
                    idf = 13.0
                if idf > 0.0:
                    idf_weight = idf / 13.0
                    #if idf < 6.0:
                    #    idf_weight *= 0.5
        if idf_weight < 0.35:
            idf_weight = 0.01
        top_kw_map[np].extend([freq_weight, idf_weight])
        # print("%s %f %f, freq:%d, max:%d, min:%d" % (np, dist, freq_weight, phrase_freq[np], max_freq, min_freq))
        # logging.debug("%s %f %f, freq:%d, max:%d, min:%d, idf:%f" % (np, dist, freq_weight, phrase_freq[np], max_freq, min_freq, idf_weight))
        final_score = dist*freq_weight*idf_weight
        if final_score < 0.075 and phrase_freq[np] == 1:
            final_score = 0.0

        if len(np) < 2:
            final_score = 0.0
        top_dist[idx] = (np, final_score)

    top_sorted = sorted(top_dist, key=lambda x:x[1], reverse=True)
    final_kw_list = []
    for kw,score in top_sorted:
        final_kw_list.append((kw, (top_kw_map[kw][0], score, top_kw_map[kw][1], top_kw_map[kw][2])))
    if check == True:
        return final_kw_list,sent_embeddings, candidate_embeddings_list, text_obj
    else:
        return final_kw_list

    #"""save for debug"""
    #check_dir = "/search/odin/liruihong/keyword-project/check_data/"
    #check_fp = open(check_dir + "tencent_plus_check_L0_pos_tfidf", "wb")
    #pickle.dump(check_fp, (sent_embeddings, candidate_embeddings_list, text_obj))
    #check_fp.close()



