# coding=utf-8
import re
import os
import sys
import numpy as np
from gensim.models import word2vec

PROC_PATH = "processed_posts/"
CUT_PATH = "cut_posts/"
SAMPLE_BEGIN = 0
SAMPLE_END = 5460

def sort_file_name_as_int(element) :
    if not element.endswith(".txt") :
        print(element, " <-- This is not a txt file. wtf did you mess out?")
        sys.exit(0)
    return int(element[0 : len(element) - 4])

def get_pagename_list(path) :
    print("\nprocessing path names...\n")
    l = os.listdir(path)
    return sorted(l, key = sort_file_name_as_int)

PAGENAME_LIST = get_pagename_list(PROC_PATH)

W2V_BY_VOCAB = True # if False: Create w2v model by each character

START_MARK = 'š'

USE_ENDING_MARK = True
ENDING_MARK = "ê"

W2V_MIN_COUNT_BY_VOCAB = 6
W2V_MIN_COUNT_BY_CHAR = 3
W2V_ITER = 6
WV_SIZE = 200

w2v_train_list = []

def get_train_data() :
    '''
    return page_list, total_word_count
    '''
    total_word_count = 0
    page_list = []
    print("fetching all post...")
    for count, pagename in enumerate(PAGENAME_LIST[SAMPLE_BEGIN : SAMPLE_END]) :
        if W2V_BY_VOCAB :
            line_list = open(CUT_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
        else :
            line_list = open(PROC_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
            
        # get words from this page
        if W2V_BY_VOCAB :
            this_page_words = [START_MARK]
        else :
            this_page_words = START_MARK
        for i, line in enumerate(line_list) :
            if re.match(r"[\s\.\-/]{4,}", line) : continue # ignore morse code
            if re.match(r"hstwproject", line) : continue # ignore meta text
            if re.match(r"^※", line) : continue # ingore translator's notes
            if re.match(r"^(.{1,12})\n", line) : continue # ignore texts in images
            if line == "\n" : continue 
            if W2V_BY_VOCAB : line = line.split() + ['\n']
            this_page_words += line
        # if this page is too short : ignore
        if len(this_page_words) < 3 :
            continue
        total_word_count += len(this_page_words)
        # put ending character at the end of a page
        if USE_ENDING_MARK :
            if W2V_BY_VOCAB :
                this_page_words += [ENDING_MARK]
            else :
                this_page_words += ENDING_MARK
        page_list.append(this_page_words)
    return page_list, total_word_count

def make_new_w2v(page_list) :
    word_model_name = "myword2vec_by_word.model" if W2V_BY_VOCAB else "myword2vec_by_char.model"
    word_model = word2vec.Word2Vec(page_list, iter = W2V_ITER, sg = 1, size = WV_SIZE, window = 6, workers = 4, min_count = (W2V_MIN_COUNT_BY_VOCAB if W2V_BY_VOCAB else W2V_MIN_COUNT_BY_CHAR))
    word_model.save(word_model_name)
    print("\nvector size: ", WV_SIZE, "\nvocab size: ", word_model.wv.syn0.shape[0])
    print(word_model.wv.most_similar("貓", topn = 10))
    
if __name__ == "__main__" :
    p, c = get_train_data()
    print("total word count:", c)
    make_new_w2v(p
    )
    print("done.")
    
