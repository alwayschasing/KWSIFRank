#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
import nltk
from model import input_representation
import thulac
import re
#GRAMMAR1 is the general way to extract NPs

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR_zh = """  NP:
        {<n.*|a|uw|i|j|x>*<n.*|uw|x>|<x|j><-><m|q>} # Adjective(s)(optional) + Noun(s)"""

GRAMMAR_sogou = """  NP:
        {<uw>} # Adjective(s)(optional) + Noun(s)"""

def extract_candidates(tokens_tagged, no_subset=False, kw_dict=None):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR_zh)  # Noun phrase parser
    #np_parser = nltk.RegexpParser(GRAMMAR_sogou)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ''.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate


def extract_candidates_indict(tokens_tagged, kw_dict):
    count = 0
    keyphrase_candidate = []
    for token,pos in tokens_tagged:
        token = token.lower()
        if token in kw_dict:
            start_end = (count, count + 1)
            keyphrase_candidate.append((token, start_end))
        #elif re.match("n.*", pos):
        #    start_end = (count, count + 1)
        #    keyphrase_candidate.append((token, start_end))
        count += 1
    return keyphrase_candidate

def extract_candidates_withpos(tokens_tagged, kw_dict):
    count = 0
    keyphrase_candidate = []
    for token,pos in tokens_tagged:
        token = token.lower()
        if token in kw_dict:
            start_end = (count, count + 1)
            keyphrase_candidate.append((token, start_end))
        elif re.match("n.*", pos):
            if len(token) < 2:
                # 过滤单字
                continue
            if len(token) == 2 and token.encode("utf-8").isalnum() == True:
                continue
            # print("[check pos candidate]%s" %(token))
            start_end = (count, count + 1)
            keyphrase_candidate.append((token, start_end))
        count += 1
    return keyphrase_candidate

def extract_candidates_incutdict(tokens_tagged, cut_kw_dict):
    keyphrase_candidate = []
    count = 0
    num = len(tokens_tagged)
    i = 0
    while i < num:
        token,pos = tokens_tagged[i]
        if token in cut_kw_dict:
            length = 1 
            tmp_dict = cut_kw_dict[token]
            match_length = 1
            if "is_leaf" in tmp_dict:
                is_match = True
            else:
                is_match = False

            for j in range(i + 1, num):
                tmp_token,tmp_pos = tokens_tagged[j]                 
                if tmp_token in tmp_dict:
                    length += 1        
                    tmp_dict = tmp_dict[tmp_token]
                    if "is_leaf" in tmp_dict:
                        is_match = True
                        match_length = length
                else:
                    break
            
            if is_match:
                candidate_kw = ""
                for k in range(0, match_length):
                    candidate_kw += tokens_tagged[i + k][0]
                start_end = (i, i + match_length)
                keyphrase_candidate.append((candidate_kw, start_end))
                i += match_length
            else:
                i += 1
        else:
            i += 1
    return keyphrase_candidate


if __name__ == '__main__':
    #This is an example.
    zh_model = thulac.thulac(model_path=r'../auxiliary_data/thulac.models/',user_dict=r'../auxiliary_data/user_dict.txt')
    sent = "以BERT为代表的自然语言预训练模型（Pre-trained Language Model）的出现使自然语言的各个任务领域的效果都得到大幅地提升。"
    ito = input_representation.InputTextObj(text=sent,zh_model=zh_model)
    keyphrase_candidate = ito.keyphrase_candidate
    for kc in keyphrase_candidate:
        print(kc)
