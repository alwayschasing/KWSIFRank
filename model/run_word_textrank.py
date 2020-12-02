#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
from WordTextRank import WordTextRank


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


def extract_words_textrank(input_file, output_file):
    word_tr = WordTextRank()
    docids, texts = load_articles(input_file)
    wfp = open(output_file, "w", encoding="utf-8")
    for idx, text in enumerate(texts):
        docid = docids[idx]
        words_textrank = word_tr.analyze(text, window=10)
        write_line = ["%s:%f" % (item.word, item.weight) for item in words_textrank]
        wfp.write("%s\t%s\n" % (docid, "\t".join(write_line)))
    wfp.close()


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_words_textrank(input_file, output_file)
