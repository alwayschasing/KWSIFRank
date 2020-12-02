#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac
import time
import jieba.analyse
import os
import csv
import logging
from multiprocessing import Process,Lock
logging.basicConfig(level=logging.DEBUG, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

user_dict_file=r'./auxiliary_data/keyword_vocab_final'
#user_dict_file=r'./auxiliary_data/user_dict.txt'
model_file = r'./auxiliary_data/zhs.model/'
ELMO = word_emb_elmo.WordEmbeddings(model_file, cuda_device=5)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=user_dict_file)
elmo_layers_weight = [0.0, 1.0, 0.0]

def load_user_dict(user_dict_file):
    user_dict = set()
    with open(user_dict_file, "r", encoding="utf-8") as fp:
        for line in fp:
            word = line.strip().split('\t')[0]
            word = word.lower()
            user_dict.add(word)
    return user_dict


def extract_keyword(text, plus=False, topk=15, kwdict=None, cut_dict=False):
    if plus == False:
        keyphrases = SIFRank(text, SIF, zh_model, N=topk,elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, cut_dict=cut_dict)
    else:
        keyphrases = SIFRank_plus(text, SIF, zh_model, N=topk, elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, cut_dict=cut_dict)
    return keyphrases

def load_articles(input_file):
    docids = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            title = parts[0]
            content = parts[1]
            docid = parts[2]
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
    user_dict_file=r'./auxiliary_data/keyword_vocab_final'
    user_dict = load_user_dict(user_dict_file)
    zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=user_dict_file)
    #docids, texts = load_articles(article_file)
    docids, texts = load_tencent_articles(article_file)
    wfp = open(output_file, "w", encoding="utf-8")
    for idx, text in enumerate(texts):
        #if idx > 10:
        #    break
        docid = docids[idx]
        logging.info("%s" % (docid))
        st = time.time()
        text_obj = text
        keywords = extract_keyword(text_obj, plus=True, topk=15, kwdict=user_dict, cut_dict=False)
        ed = time.time()
        cost = int((ed - st) * 1000)
        print("%s time_cost:%d ms" % (docid, cost))
        #keywords = [(kw,score) for kw,score in keywords if kw in user_dict]
        writer_line = " ".join(["%s:%f" % (kw, score) for kw,score in keywords])
        wfp.write("%s\t%s\n" % (docid, writer_line))
    wfp.close()


if __name__ == "__main__":
    #article_file = "/search/odin/liruihong/article_data/articles_test.tsv"
    article_file = "/search/odin/liruihong/keyword-project/data/tencent_data/tencent_articles.tsv"
    output_file = "/search/odin/liruihong/keyword-project/output_data/tencent_kw_plus_freqall_pos_top15"
    extract_article_kw(article_file, output_file)

