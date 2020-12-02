#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

from embeddings import sent_emb_sif, word_emb_elmo
from model.method_test import SIFRank, SIFRank_plus
import thulac
import pickle
import jieba.analyse
import os
import csv
import logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

user_dict_file=r'./auxiliary_data/keyword_vocab_final'
#user_dict_file=r'/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_clean'
#user_dict_file=r'./auxiliary_data/user_dict.txt'
#user_dict_file=None
model_file = r'./auxiliary_data/zhs.model/'
ELMO = word_emb_elmo.WordEmbeddings(model_file, cuda_device=6)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=user_dict_file, seg_only=False)
elmo_layers_weight = [1.0, 0.0, 0.0]

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


def extract_keyword(text, plus=False, topk=15, kwdict=None, kw_info=None, cut_dict=False, seg_only=True, use_pos=False, check=False):
    if plus == False:
        keyphrases = SIFRank(text, SIF, zh_model, N=topk,elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, cut_dict=cut_dict)
    else:
        keyphrases = SIFRank_plus(text, SIF, zh_model, N=topk, elmo_layers_weight=elmo_layers_weight,
                                kwdict=kwdict, kw_info=kw_info, cut_dict=cut_dict, seg_only=seg_only, use_pos=use_pos, logger=logger, check=check)
    return keyphrases

def load_articles(input_file):
    docids = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                print("[parts error] less 3")
                continue

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

def load_kw_idf(kw_idf_file, encoding="utf-8"):
    kw_info = dict()
    with open(kw_idf_file, "r", encoding=encoding) as fp:
        for line in fp:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            kw, df ,idf = parts[0], int(parts[1]), float(parts[2])
            kw_info[kw] = (idf, df)
    return kw_info


def extract_article_kw(article_file, output_file):
    #docids, texts = load_articles(article_file)
    docids, texts = load_tencent_articles(article_file)
    wfp = open(output_file, "w", encoding="utf-8")
    #cut_dict_file=r'/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_vocab_final_cut'
    #user_dict = load_cut_dict(cut_dict_file)
    user_dict_file=r'./auxiliary_data/keyword_vocab_final'
    user_dict = load_user_dict(user_dict_file)
    #kw_info_file = r'/search/odin/liruihong/keyword-project/config_data/ret_item_info'
    kw_info_file = r'/search/odin/liruihong/keyword-project/item_df_statistic/item_df.tsv'
    #kw_info = load_kw_info(kw_info_file, encoding="utf-8")
    kw_info = load_kw_idf(kw_info_file, encoding="utf-8")

    """save for debug"""
    check_dir = "/search/odin/liruihong/keyword-project/check_data/"
    check_fp = open(check_dir + "tencent_plus_check_L0_pos_tfidf", "wb")
    check_data = {}
    for idx, text in enumerate(texts):
        if idx > 200:
            break
        docid = docids[idx]
        logging.info("%s" % (docid))
        #keywords = extract_keyword(text, plus=True, topk=20, kwdict=user_dict, kw_info=kw_info, cut_dict=False, seg_only=False, use_pos=True)
        keywords, sent_embeddings, candidate_embeddings_list, text_obj = extract_keyword(text, plus=True, topk=20, kwdict=user_dict, kw_info=kw_info, cut_dict=False, seg_only=False, use_pos=True, check=True)
        check_data[docid] = {
            "sent_embs":sent_embeddings,
            "candidate_embs":candidate_embeddings_list,
            "text_obj":text_obj
        }

        #print(keywords)
        #keywords = [(kw,score) for kw,score in keywords if kw in user_dict]
        #keywords = keywords[0:10]
        #writer_line = " ".join(["%s:%f" % (kw, score) for kw,score in keywords])
        writer_line = " ".join(["%s:%s" % (kw, ",".join(["%f" %(x) for x in score])) for kw,score in keywords])
        wfp.write("%s\t%s\n" % (docid, writer_line))
    wfp.close()

    pickle.dump(check_data, check_fp)
    check_fp.close()


if __name__ == "__main__":
    #article_file = "/search/odin/liruihong/article_data/articles_test.tsv"
    article_file = "/search/odin/liruihong/keyword-project/data/tencent_data/tencent_articles.tsv"
    output_file = "/search/odin/liruihong/keyword-project/output_data/tencent_kw_plus_L0_pos_tfidf.tsv"
    #article_file = "/search/odin/liruihong/keyword-project/annotate_data/click_articles.tsv"
    #output_file = "/search/odin/liruihong/keyword-project/annotate_data/clickdoc_kw_plus_L0_pos_tfidf.tsv"
    extract_article_kw(article_file, output_file)
    #load_cut_dict('/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_vocab_final_cut')

