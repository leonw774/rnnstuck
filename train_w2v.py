#py coding=utf-8
import re
import os
import sys
import random
import numpy as np
from configure import *
from gensim.models import word2vec

PROC_PATH = "processed_posts/"
CUT_PATH = "cut_posts/"
USED_PATH = CUT_PATH if W2V_BY_VOCAB else PROC_PATH
SAMPLE_BEGIN = 0
SAMPLE_END = -1

def sort_file_name_as_int(element) :
    if not element.endswith(".txt") :
        print(element, " <-- This is not a txt file. wtf did you mess out?")
        sys.exit(0)
    return int(element[0 : len(element) - 4])

def get_pagename_list(path) :
    print("\nprocessing path names...\n")
    l = os.listdir(path)
    return sorted(l, key = sort_file_name_as_int)

PAGENAME_LIST = get_pagename_list(USED_PATH)

def get_train_data(page_amount = None, word_min = 1, word_max = None, line_min = 1, line_max = None) :
    '''
    return page_list, total_word_count
    '''
    total_word_count = 0
    page_list = []
    pagename_list = PAGENAME_LIST[SAMPLE_BEGIN : SAMPLE_END]
    if page_amount:
        pagename_list = random.sample(PAGENAME_LIST[SAMPLE_BEGIN : SAMPLE_END], page_amount)
    
    for _, pagename in enumerate(pagename_list) :
        line_list = open(USED_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
        
        if len(line_list) < line_min : continue
        if line_max:
            if len(line_list) >= line_max : line_list = line_list[ : line_max]
        
        # get words from this page
        if USE_START_MARK :
            this_page_words = [START_MARK] if W2V_BY_VOCAB else START_MARK
        else :
            this_page_words = [] if W2V_BY_VOCAB else ""
        
        for i, line in enumerate(line_list) :
            line = line.strip()
            if (re.match(r"[\s\.\-/]{5,}", line) or # ignore morse code
                re.match(r"hstwproject|zhhomestuck", line) or # ignore meta text
                re.match(r"(^※)", line) or # ingore translator's notes
                re.match(r"^\([ 圖片影片中的純文字翻譯下收]{4,20}\)", line) # ignore texts in images
               ) :
                #print(line)
                continue
            #line = re.sub(r"※[0-9 ]", "", line) # delete translator's notes in content
            if W2V_BY_VOCAB : line = line.split() + ['\n']
            this_page_words += line
            if word_max:
                if len(this_page_words) >= word_max : break
        # because there is a line break at the end
        this_page_words = this_page_words[ : -1]
        # if this page is too short : ignore
        if len(this_page_words) < word_min : continue
        total_word_count += len(this_page_words)
        # put ending character at the end of a page
        if USE_ENDING_MARK :
            if W2V_BY_VOCAB :
                this_page_words.append(ENDING_MARK)
            else :
                this_page_words += ENDING_MARK
        page_list.append(this_page_words)
    return page_list, total_word_count

def make_new_w2v(page_list, show_result = False) :
    print("make word model...")
    word_model = word2vec.Word2Vec(page_list, iter = W2V_ITER, sg = 1, size = WV_SIZE, window = 6, workers = 4, min_count = W2V_MIN_COUNT)
    word_model.save(W2V_MODEL_NAME)
    if show_result :
        print("vector size: ", WV_SIZE, "\nvocab size: ", word_model.wv.syn0.shape[0])
        if W2V_BY_VOCAB :
            print("遊戲\n", word_model.wv.most_similar("遊戲", topn = 10))
        else :
            print("貓\n", word_model.wv.most_similar("貓", topn = 10))
    print("done.")
    return word_model
    
if __name__ == "__main__" :
    p, c = get_train_data()
    print("total word count:", c)
    make_new_w2v(p, show_result = True)
    print("done.")
    
