#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

from model import extractor
import thulac
import logging
import time
import re
from model import util
logger = util.get_logger(__name__, debug=1)
stopwords_file="/search/odin/liruihong/word2vec_embedding/hit_stopwords.txt"
stopwords_dict = set(util.load_vocab(stopwords_file))

sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
def sentence_split(text, sent_delimiters=sentence_delimiters):
    res = [text]
    for sep in sent_delimiters:
        text, res = res, []
        for seq in text:
            res += seq.split(sep)
    res = [s.strip() for s in res if len(s.strip()) > 0]
    return res

def segment_sentences(segmenter, sentences, lower=True):
    res = []
    for sent in sentences:
        res.append(segmenter.cut(sent))
    return res

def clean_sentence_words(sentence_words, stopwords=stopwords_dict):
    new_sentences_words = []
    for words in sentence_words:
        new_words = []
        for w in words:
            w = (w[0].lower(), w[1])
            if w[0] in stopwords:
                continue
            new_words.append(w)
        new_sentences_words.append(new_words)
    return new_sentences_words

class InputTextObj:
    def __init__(self, zh_model, text="", kw_dict=None, cut_dict=False, seg_only=False, use_pos=False, user_dict=None, logger=logging.getLogger()):
        """
        :param is_sectioned: If we want to section the text.
        :param zh_model: the pipeline of Chinese tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'n', 'np', 'ns', 'ni', 'nz','a','d','i','j','x','g'}
        self.tokens = []
        self.tokens_tagged = []
        self.sentences = []
        # self.tokens = zh_model.cut(text)
        st = time.time()
        if len(text) > 5000:
            text = text[0:5000]

        self.sentences = sentence_split(text)
        sentence_words = segment_sentences(zh_model, self.sentences)

        word_pos = []
        for words in sentence_words:
            word_pos.extend(words) 
        
        self.sentence_words = []
        for words in sentence_words:
            words = [x[0] for x in words]
            self.sentence_words.append(words)
        sentence_words = clean_sentence_words(sentence_words)

        # word_pos = zh_model.cut(text)
        ed = time.time()
        cost = int((ed - st)*1000)
        #print("cut_cost:%dms" %(cost))

        def clean_tokens(word_pos):
            new_tokens = []
            new_tokens_tagged = []
            for w,p in word_pos:
                match_res = re.match("《(.*)》", w)
                if match_res is not None:
                    w = match_res.group(1)
                w = re.sub(",","", w)
                new_tokens.append(w)
                new_tokens_tagged.append((w,"n"))
            return new_tokens, new_tokens_tagged

        self.tokens, self.tokens_tagged = clean_tokens(word_pos)
        assert len(self.tokens) == len(self.tokens_tagged)

        for i, token in enumerate(self.tokens):
            if token.lower() in stopwords_dict:
                self.tokens_tagged[i] = (token, "u")
            if token == '-':
                self.tokens_tagged[i] = (token, "-")

        if use_pos == False:
            self.keyphrase_candidate = extractor.extract_candidates_indict(self.tokens_tagged, kw_dict)
        else:
            self.keyphrase_candidate = extractor.extract_candidates_withpos(self.tokens_tagged, kw_dict)

        cut_res = ["%s:%s" % (a,b) for (a,b) in self.tokens_tagged]
        print_line = ["%s:%d,%d" % (w,a,b) for w,(a,b) in self.keyphrase_candidate]
        logger.debug("[check_cut]len:%d  %s" % (len(cut_res), " ".join(cut_res)))
        logger.debug("[check_candidate] %s" % (" ".join(print_line)))

# if __name__ == '__main__':
#     text = "以BERT为代表的自然语言预训练模型（Pre-trained Language Model）的出现使自然语言的各个任务领域的效果都得到大幅地提升。"
#     zh_model = thulac.thulac(model_path=r'../auxiliary_data/thulac.models/')
#     out1 = zh_model.cut(text,text=False)
#
