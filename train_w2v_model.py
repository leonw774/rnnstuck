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

W2V_BY_VOCAB = True # False: Create w2v model by each character
USE_ENDING_MARK = True
ENDING_MARK_WORD = "<e>"
ENDING_MARK_CHAR = '\0'

W2V_MIN_COUNT = 6
W2V_ITER = 6
WV_SIZE = 200

w2v_train_list = []

def make_new_w2v() :
    total_word_count = 0
    page_list = []
    print("fetching all post...")
    for count, pagename in enumerate(PAGENAME_LIST[SAMPLE_BEGIN : SAMPLE_END]) :
        if W2V_BY_VOCAB :
            line_list = open(CUT_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
        else :
            line_list = open(PROC_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
            
        # get words from this page
        this_page_word_list = []
        for i, line in enumerate(line_list) :
        
            if re.match(r"[\s\.\-/]{4,}", line) : continue # ignore morse code
            if line == "\n" : continue 
            if W2V_BY_VOCAB : line = line.split() + ['\n']
            
            total_word_count += len(line)
            this_page_word_list += line
        
        # if this page is too short : ignore
        if len(this_page_word_list) < 3 :
            continue    
        # put ending character at the end of a page
        if USE_ENDING_MARK :
            if W2V_BY_VOCAB :
                this_page_word_list.append(ENDING_MARK_WORD)
            else :
                this_page_word_list += ENDING_MARK_CHAR
        page_list.append(this_page_word_list)
    
    print("total word count:", total_word_count)

    word_model_name = "myword2vec_by_word.model" if W2V_BY_VOCAB else "myword2vec_by_char.model"
    word_model = word2vec.Word2Vec(page_list, iter = W2V_ITER, sg = 1, size = WV_SIZE, window = 6, workers = 4, min_count = (W2V_MIN_COUNT if W2V_BY_VOCAB else 3))
    word_model.save(word_model_name)
    print("\nvector size: ", WV_SIZE, "\nvocab size: ", word_model.wv.syn0.shape[0])
    print(word_model.wv.most_similar("貓", topn = 10))
    
if __name__ == "__main__" :
    make_new_w2v()
    print("done.")
    
