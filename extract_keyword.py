#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

import scipy
from embeddings import sent_emb_sif, word_emb_elmo, word2vec_emb
from model.method_textrank import SIFRank, SIFRank_plus
import thulac
import pickle
import time
import jieba.analyse
import os
import csv
import logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

user_dict_file=r'./auxiliary_data/keyword_vocab_final'
#user_dict_file=r'./auxiliary_data/user_dict.txt'
model_file = r'./auxiliary_data/zhs.model/'
# ELMO = word_emb_elmo.WordEmbeddings(model_file, cuda_device=7)
# SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
seg_only = True
zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=user_dict_file, seg_only=seg_only)
elmo_layers_weight = [0.5, 0.5, 0.0]
kw_info_file=r'/search/odin/liruihong/keyword-project/config_data/ret_item_info'


def load_kw_info(kw_info_file, encoding="utf-8"):
    kw_info = dict()
    with open(kw_info_file, "r", encoding=encoding) as fp:
        for line in fp:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            kw, qv, df, idf = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
            kw_info[kw] = (idf,df,qv)
    return kw_info

def load_user_dict(user_dict_file):
    user_dict = set()
    with open(user_dict_file, "r", encoding="utf-8") as fp:
        for line in fp:
            word = line.strip().split('\t')[0]
            word = word.lower()
            user_dict.add(word)
    return user_dict

def extract_keyword(text, SIF, zh_model, elmo_layers_weight, plus=False, topk=15, kwdict=None, kw_info=None, cut_dict=False, seg_only=True, user_dict=None):
    if plus == False:
        keyphrases = SIFRank(text, SIF, zh_model, N=topk,elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, kw_info=kw_info, cut_dict=cut_dict, seg_only=True)
    else:
        keyphrases = SIFRank_plus(text, SIF, zh_model, N=topk, elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, kw_info=kw_info, cut_dict=cut_dict, seg_only=True, user_dict=user_dict)
    return keyphrases

def load_articles(input_file):
    docids = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            docid = parts[0]
            title = parts[1]
            content = parts[2]
            text = title + " " + content
            docids.append(docid)
            texts.append(text)
    return docids, texts


def load_tencent_articles(input_file):
    docids = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            docid = parts[0]
            title = parts[-2]
            content = parts[-1]
            text = title + " " + content
            docids.append(docid)
            texts.append(text)
    return docids, texts


def extract_article_kw(article_file, output_file):
    docids, texts = load_articles(article_file)
    #docids, texts = load_tencent_articles(article_file)
    wfp = open(output_file, "w", encoding="utf-8")
    user_dict_file=r'./auxiliary_data/keyword_vocab_final'
    kw_info_file=r'/search/odin/liruihong/keyword-project/config_data/ret_item_info'
    model_file = r'./auxiliary_data/zhs.model/'
    gpu_id = 0

    kwdict = load_user_dict(user_dict_file)
    kw_info = load_kw_info(kw_info_file, encoding="gbk")

    user_dict = load_user_dict("/search/odin/liruihong/keyword-project/keywords_embrank/config_data/articles_7d_vocab.txt")

    seg_only = True
    # ELMO = word_emb_elmo.WordEmbeddings(model_file, cuda_device=gpu_id)
    # SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
    w2v = word2vec_emb.Word2VecEmbeddings("/search/odin/liruihong/keyword-project/keywords_embrank/config_data/articles_7d_word2vec.txt")
    SIF = sent_emb_sif.SentEmbeddings(w2v, lamda=1.0, embeddings_type="glove")
    # zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/', seg_only=seg_only)
    elmo_layers_weight = [1.0, 0.0, 0.0]
    plus = True
    kw_info = kw_info

    checkfp = open("/search/odin/liruihong/keyword-project/SIFRank_zh/check_w2vsif_model.bin","wb")
    check_data = {}
    for idx, text in enumerate(texts):
        #if idx >= 200:
        #    break
        docid = docids[idx]
        logging.info("%s" % (docid))
        st = time.time()
        keywords = extract_keyword(text, SIF, zh_model, elmo_layers_weight, plus=plus,
                                    topk=20, kwdict=kwdict, kw_info=kw_info, cut_dict=False, seg_only=seg_only, user_dict=user_dict)

        if isinstance(keywords, tuple):
            check_item = keywords[1]
            check_data[docid] = check_item
            keywords = keywords[0]
        ed = time.time()
        cost = int((ed - st) * 1000)
        print("%s time_cost:%d ms" % (docid, cost))
        #keywords = [(kw,score) for kw,score in keywords if kw in user_dict]
        writer_keywords = " ".join(["%s:%f" % (kw,score) for kw,score in keywords])
        wfp.write("%s\t%s\n" % (docid, writer_keywords))

    pickle.dump(check_data, checkfp)
    checkfp.close()
    wfp.close()


if __name__ == "__main__":
    article_file = "/search/odin/liruihong/keyword-project/input_data/test_articles.tsv"
    output_file = "/search/odin/liruihong/keyword-project/output_data/word2vec_sifrank.tsv"
    extract_article_kw(article_file, output_file)

