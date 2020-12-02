# -*- encoding:utf-8 -*-
import os
import math
import numpy as np
import sys
import logging

sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['n', 'np', 'ns', 'ni', 'nz', 'a', 'id', 'uw', 'v', 'nr', 't', 'f', 's', 'i', 'j', 'x']

def get_logger(name, debug=0):
    if debug == 1:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel,
                        format="[%(levelname).1s %(asctime)s] %(message)s",
                        datefmt="%Y-%m-%d_%H:%M:%S")
    logger = logging.getLogger(name)
    return logger


class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_vocab(input_file):
    vocab = []
    fp = open(input_file, "r", encoding="utf-8")
    for line in fp:
        word = line.strip().split('\t')[0]
        vocab.append(word)
    return vocab

if __name__ == '__main__':
    pass
