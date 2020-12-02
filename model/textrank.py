#-*- encoding:utf-8 -*-
import networkx as nx
import numpy as np
from model import util
logger = util.get_logger(__name__)

class AttrDict(dict):
    """Dict that can get attribute by dot"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def combine(word_list, window=2):
    """构造在window下的单词组合，用来构造单词之间的边。

    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。
    """
    if window < 2:
        window = 2
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r


def get_textrank(doc_sents, window=10, pagerank_config={'alpha': 0.85,}):
    word_index = {}
    index_word = {}
    _vertex_source = doc_sents 
    _edge_source = doc_sents 
    words_number = 0
    for word_list in _vertex_source:
        for word in word_list:
            if word not in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

    graph = np.zeros((words_number, words_number))
    # check_data = {"word_index":word_index,
    #               "graph":graph}

    for word_list in _edge_source:
        for w1, w2 in combine(word_list, window):
            if w1 in word_index and w2 in word_index:
                index1 = word_index[w1]
                index2 = word_index[w2]
                graph[index1][index2] += 1.0
                graph[index2][index1] += 1.0

    nx_graph = nx.from_numpy_matrix(graph)
    try:
        scores = nx.pagerank(nx_graph, **pagerank_config)          # this is a dict
    except Exception as e:
        print(e)
        return []
    # sorted_scores = sorted(
    #     scores.items(),
    #     key=lambda item: item[1],
    #     reverse=True)
    # for index, score in sorted_scores:
    #     # item = AttrDict(word=index_word[index], weight=score)
    #     item = (index_word[index], score)
    #     sorted_words.append(item)
    item_dict = {}
    # max_num = 0.0; min_num = 1.0
    # for index, score in scores.items():
    #     if score > max_num:
    #         max_num = score
    #     if score < min_num:
    #         min_num = score

    # denominator = max_num - min_num 
    for index, score in scores.items():
        # items.append(index_word[index], score) 
        # norm_score = (score - min_num) / denominator 
        # item_dict[index_word[index]] = norm_score
        item_dict[index_word[index]] = score
    # logger.info("[check textrank] max_s:%f, min_s:%f, denominator:%f, %s" % (max_num, min_num, denominator, item_dict))
    logger.info("[check textrank] %s" % (item_dict))
    return item_dict


if __name__ == '__main__':
    pass
