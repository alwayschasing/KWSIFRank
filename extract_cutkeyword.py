#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac
import jieba.analyse
import os
import csv
import logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

#user_dict_file=r'./auxiliary_data/keyword_vocab_final'
#user_dict_file=r'./auxiliary_data/user_dict.txt'
user_dict_file=None
model_file = r'./auxiliary_data/zhs.model/'
ELMO = word_emb_elmo.WordEmbeddings(model_file, cuda_device=4)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=user_dict_file)
elmo_layers_weight = [0.0, 1.0, 0.0]

def load_cut_dict(user_dict_file):
    trie_dict = dict()
    with open(user_dict_file, "r", encoding="utf-8") as fp:
        for line in fp:
            cut_parts = line.strip().split(' ')
            num = len(cut_parts)
            tmp_dict = trie_dict
            for i in range(num):
                p = cut_parts[i]
                if p in tmp_dict:
                    tmp_dict = tmp_dict[p]
                else:
                    tmp_dict[p] = dict()
                    tmp_dict = tmp_dict[p]
                if i == num - 1:
                    tmp_dict.update({"is_leaf":1})
    return trie_dict


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
    #docids, texts = load_articles(article_file)
    docids, texts = load_tencent_articles(article_file)
    wfp = open(output_file, "w", encoding="utf-8")
    cut_dict_file=r'/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_vocab_final_cut'
    user_dict = load_cut_dict(cut_dict_file)
    for idx, text in enumerate(texts):
        docid = docids[idx]
        logging.info("%s" % (docid))
        keywords = extract_keyword(text, plus=False, topk=15, kwdict=user_dict, cut_dict=True)
        #print(keywords)
        #keywords = [(kw,score) for kw,score in keywords if kw in user_dict]
        #keywords = keywords[0:10]
        writer_line = " ".join(["%s:%f" % (kw, score) for kw,score in keywords])
        wfp.write("%s\t%s\n" % (docid, writer_line))
    wfp.close()


if __name__ == "__main__":
    #article_file = "/search/odin/liruihong/article_data/articles_test.tsv"
    article_file = "/search/odin/liruihong/keyword-project/data/tencent_data/tencent_articles.tsv"
    output_file = "/search/odin/liruihong/keyword-project/output_data/tencent_kw_cut.tsv"
    extract_article_kw(article_file, output_file)
    #load_cut_dict('/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_vocab_final_cut')

