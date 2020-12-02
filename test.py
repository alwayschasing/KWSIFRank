#! /usr/bin/env python
# -*- coding: utf-8 -*-
import thulac

def load_cut_dict(user_dict_file):
    trie_dict = dict()
    with open(user_dict_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for idx,line in enumerate(lines):
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
    
    #for k,v in trie_dict.items():
    #    print("%s\t%s" %(k,v))
    return trie_dict

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
                    if "is_leaf" in tmp_dict:
                        is_match = True
                        match_length = length
                    tmp_dict = tmp_dict[tmp_token]
                else:
                    break
            
            if is_match:
                candidate_kw = ""
                for k in range(0, match_length):
                    candidate_kw += tokens_tagged[i + k][0]
                start_end = (i, i + match_length - 1)
                keyphrase_candidate.append((candidate_kw, start_end))
                print("%s, %s" %(candidate_kw, start_end))
                i += match_length
            else:
                i += 1
        else:
            i += 1
    return keyphrase_candidate

def load_text(input_file, target_docid):
    text = ""
    with open(input_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            docid = parts[0]
            if docid == target_docid:
                title = parts[3]
                content = parts[4]
                text = title + " " + content
                break
    return text
            


if __name__ == "__main__":
    #zh_model = thulac.thulac(model_path=r'./auxiliary_data/thulac.models/',user_dict=None)
    #docid = "txnews_20200818A0RP1R"
    #input_file = "/search/odin/liruihong/keyword-project/data/tencent_data/tencent_articles.tsv"
    #text = load_text(input_file, docid)
    #if text == "":
    #    raise ValueError("text null")
    #word_pos = zh_model.cut(text)
    #word_pos = [(word_pos[0],word_pos[1]) for word_pos in word_pos]
    #word_pos = word_pos[0:30]
    #words = [x[0] for x in word_pos]
    #print(" ".join(words))
    word_pos = [("中", "n"), ("银","n"), ("富","n"), ("登","n"), ("村镇","n"), ("银行","n"), ("股份","n"), ("有限公司","n")]
    cut_kw_dict = load_cut_dict('/search/odin/liruihong/keyword-project/data/keywords_vocab/keyword_vocab_final_cut')
    extract_candidates_incutdict(word_pos, cut_kw_dict)