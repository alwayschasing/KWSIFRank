#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac
import time
import os
import csv
from model import util
import logging
import multiprocessing as mp
from multiprocessing import Process,Lock,Queue,Value
logger = util.get_logger(__name__, debug=1)

#user_dict_file=r'./auxiliary_data/keyword_vocab_final'
##user_dict_file=r'./auxiliary_data/user_dict.txt'
#model_file = r'./auxiliary_data/zhs.model/'
#ELMO = word_emb_elmo.WordEmbeddings(model_file, cuda_device=5)
#SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
#zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=user_dict_file)
#elmo_layers_weight = [0.0, 1.0, 0.0]

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


def extract_keyword(text, SIF, zh_model, elmo_layers_weight, plus=False, topk=15, kwdict=None, kw_info=None, cut_dict=False, seg_only=True):
    if plus == False:
        keyphrases = SIFRank(text, SIF, zh_model, N=topk,elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, kw_info=kw_info, cut_dict=cut_dict, seg_only=True)
    else:
        keyphrases = SIFRank_plus(text, SIF, zh_model, N=topk, elmo_layers_weight=elmo_layers_weight, kwdict=kwdict, kw_info=kw_info, cut_dict=cut_dict, seg_only=True)
    return keyphrases

def load_articles(input_file):
    docids = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for idx,line in enumerate(fp):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                print("[parts error]less 3")
                continue
            docid = parts[0]
            title = parts[1]
            content = parts[2]
            text = title + "\t" + content
            docids.append(docid)
            texts.append(text)
            if idx < 5:
                print("[check_data]docid:%s, title:%s, content:%s" %(docid, title, content))
    return docids, texts


def load_tencent_articles(input_file):
    docids = []
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for idx,line in enumerate(fp):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            docid = parts[0]
            title = parts[3]
            content = parts[4]
            text = title + "\t" + content
            docids.append(docid)
            texts.append(text)
            if idx < 5:
                print("[check_data]docid:%s, title:%s, content:%s" %(docid, title, content))
    return docids, texts


class ExtractWorker(Process):
    def __init__(self, recv_queue, push_queue, stop_sign, worker_id, gpu_id=0, plus=True, seg_only=True, elmo_layers_weight=[0.5, 1.0, 0.5], cut_dict=False, logger=logging.getLogger()):
        super(ExtractWorker, self).__init__()
        self.user_dict_file=r'./auxiliary_data/keyword_vocab_final'
        self.cut_dict_file=r'/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_vocab_final_cut'
        self.kw_info_file=r'/search/odin/liruihong/keyword-project/config_data/ret_item_info'
        self.model_file = r'./auxiliary_data/zhs.model/'
        self.seg_only = seg_only
        self.gpu_id = gpu_id
        if cut_dict == False:
            self.user_dict = load_user_dict(self.user_dict_file)
        else:
            self.user_dict = load_cut_dict(self.user_dict_file)
        self.cut_dict = cut_dict
        self.kw_info = load_kw_info(self.kw_info_file, encoding="gbk")
        self.elmo_layers_weight = elmo_layers_weight
        self.recv_queue = recv_queue
        self.push_queue = push_queue
        self.stop_sign = stop_sign
        self.plus = plus
        self.worker_id = worker_id
        self.logger = logger

    def run(self):
        self.ELMO = word_emb_elmo.WordEmbeddings(self.model_file, cuda_device=self.gpu_id)
        self.SIF = sent_emb_sif.SentEmbeddings(self.ELMO, lamda=1.0)
        if self.cut_dict == True:
            self.zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/', seg_only=self.seg_only)
        else:
            self.zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=self.user_dict_file, seg_only=self.seg_only)
        while self.stop_sign.value == 0:
            if self.recv_queue.empty() == False:
                try:
                    data = self.recv_queue.get(True, 1)
                except Exception as e:
                    continue
                docid = data[0]
                text = data[1]
                if len(text) > 4000:
                    text = text[0:4000]
                # [title, content] = text.split('\t')
                self.logger.info("worker_process[%d] %s, len:%d" %(self.worker_id, docid, len(text)))
                keywords = extract_keyword(text, self.SIF, self.zh_model, self.elmo_layers_weight, plus=self.plus,
                                            topk=20, kwdict=self.user_dict, kw_info=self.kw_info, cut_dict=self.cut_dict, seg_only=self.seg_only)

                self.logger.info("worker_succ[%d] %s" %(self.worker_id, docid))
                self.logger.info("worker_succ[%d] %s %s" %(self.worker_id, docid, keywords))
                #self.push_queue.put([docid, title_kw, content_kw])
                self.push_queue.put([docid, keywords])

        self.logger.info("stop worker[%d]" %(self.worker_id))


def multiprocess_extract_keywords(input_file, output_file, process_num=1, gpu_ids=[0], plus=True, elmo_layers_weight=[1.0, 0.0, 0.0], cut_dict=False, seg_only=True):
    input_que = Queue()
    output_que = Queue()
    #docids, texts = load_tencent_articles(input_file)
    docids, texts = load_articles(input_file)
    total_num = len(docids)
    real_num = 0
    for i in range(total_num):
        if i >= 40:
            break
        real_num += 1
        docid = docids[i]
        text = texts[i]
        input_que.put([docid, text])

    stop_sign = Value('i', 0) # 进程间共享停止变量
    worker_list = []
    for i in range(process_num):
        gpu_idx = i % (len(gpu_ids))
        logger.info("create worker[%d] on gpu_%d" %(i, gpu_ids[gpu_idx]))
        worker = ExtractWorker(input_que, output_que, stop_sign, worker_id=i, gpu_id=gpu_ids[gpu_idx], plus=plus, seg_only=seg_only,
                               elmo_layers_weight=elmo_layers_weight, cut_dict=cut_dict, logger=logger)
        worker_list.append(worker)


    for i,worker in enumerate(worker_list):
        worker.start()
        logger.info("start worker[%d]" %(i))

    st = time.time()
    res_num = 0
    wfp = open(output_file, "w", encoding="utf-8")
    speed_st = time.time()
    speed_count = 0
    while True:
        if res_num == real_num:
            break
        if output_que.empty() == False:
            try:
                data = output_que.get(True, 1)
            except Exception as e:
                logger.error("multiprocess_extract_keywords output_que.get Exception:%s" % (e))
                continue
            docid = data[0]
            keywords = data[1]
            #title_keywords = data[1]
            #content_keywords = data[2]
            #writer_title = " ".join(["%s:%s" % (kw, ",".join(["%f" %(x) for x in score])) for kw,score in title_keywords])
            #writer_content = " ".join(["%s:%s" % (kw, ",".join(["%f" %(x) for x in score])) for kw,score in content_keywords])
            writer_keywords = "\t".join(["%s:%s" % (kw_score[0], ",".join(["%f" %(x) for x in kw_score[1:]])) for kw_score in keywords])
            #writer_line = " ".join(["%s:%f" % (k,s) for k,s in keywords])
            #wfp.write("%s\t%s\t%s\n" % (docid, writer_title, writer_content))
            logger.info("%s\t%s\n" % (docid, writer_keywords))
            wfp.write("%s\t%s\n" % (docid, writer_keywords))
            res_num += 1
            logger.info("[succ]%s" %(docid))
            speed_count += 1
            speed_ed = time.time()
            if int(speed_ed - speed_st) >= 60:
                #speed = speed_count/60
                logger.info("[check_speed]%d/minute" %(speed_count))
                speed_st = time.time()
                speed_count = 0

    wfp.close()
    ed = time.time()
    cost = int((ed - st)/60)
    logger.info("all data cost:%d minutes %d seconds" %(cost, int(ed-st)))
    stop_sign.value = 1
    for worker in worker_list:
        worker.join()
    logger.info("finish all work")


if __name__ == "__main__":
    # input_file = "/search/odin/liruihong/keyword-project/input_data/new_articles_7d.tsv"
    # output_file = "/search/odin/liruihong/keyword-project/output_data/new_articles_7d_kw.tsv"
    input_file = "/search/odin/liruihong/keyword-project/input_data/test_articles.tsv"
    output_file = "/search/odin/liruihong/keyword-project/output_data/text_articles_kw_siftr"
    multiprocess_extract_keywords(input_file, output_file, process_num=4, gpu_ids=[3,4,5,6], plus=True, elmo_layers_weight=[1.0, 0.0, 0.0], cut_dict=False, seg_only=True)

