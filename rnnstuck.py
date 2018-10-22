﻿import os
import re
import sys
import numpy as np
import jieba_zhtw as jb
import random
import h5py
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, multiply, Permute, RepeatVector, TimeDistributed
from gensim.models import word2vec

#os.environ["CUDA_VISIBLE_DEVICES"] = '1' # -1 : Use CPU; 0 or 1 : Use GPU

PROC_PATH = "processed_posts/"
CUT_PATH = "cut_posts/"

CREATE_NEW_JIEBA = False
CREATE_NEW_W2V = False
W2V_BY_EACH_WORD = True # False: Create w2v model by each character
USE_ENDING_MARK = True
ENDING_MARK = "<e>"

MIN_COUNT = 4
W2V_ITER = 10
VOCAB_SIZE = -1

SAMPLE_BEGIN = 0
SAMPLE_END = 5420
POST_LENGTH_MAX = 2700
POST_LENGTH_MIN = 6

USE_SAVED_MODEL = True
SAVE_MODEL_NAME = "rnnstuck_model.h5"
WV_SIZE = 200
RNN_UNIT = 32 # 1 core nvidia gt730 gpu: lstm(300) is limit
EPOCHS = 64
OUTPUT_NUMBER = 50
# 4000 sample +
# W2V_BY_EACH_WORD = True +
# 100 unit lstm +
# 1 epoch
#    ==> 20~25 minute


def sorting_file_name(element) :
    if not element.endswith(".txt") :
        print(element, " <-- This is not a txt file. wtf did you mess out?")
        sys.exit(0)
    return int(element[0 : len(element) - 4])

print("\nprocessing path names...\n")
POSTNAME_LIST = []
for filename in os.listdir(PROC_PATH) :
    POSTNAME_LIST.append(filename)
POSTNAME_LIST = sorted(POSTNAME_LIST, key = sorting_file_name)

### JIEBA ###
print("CREATE_NEW_JIEBA: ", CREATE_NEW_JIEBA)
if CREATE_NEW_JIEBA :
    jb.dt.cache_file = 'jieba.cache.zhtw'
    jb.load_userdict('hs_dict.dict')
    for postname in POSTNAME_LIST :
        jieba_in_string = open(PROC_PATH + postname, 'r', encoding = "utf-8-sig").read() 
        cut_post = jb.cut(jieba_in_string, cut_all = False)
        open(CUT_PATH + postname, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_post))  
### END JIEBA ###

print("\nW2V_BY_EACH_WORD: ", W2V_BY_EACH_WORD, "\nCREATE_NEW_W2V: ", CREATE_NEW_W2V, "\niter: ", W2V_ITER)

### PREPARE TRAINING DATA ###
total_word_count = 0
rnn_train_input = []
w2v_train_list = []
for count, postname in enumerate(POSTNAME_LIST[SAMPLE_BEGIN : SAMPLE_END]) :

    if W2V_BY_EACH_WORD :
        line_list = open(CUT_PATH + postname, 'r', encoding = 'utf-8-sig').readlines()
    else :
        line_list = open(PROC_PATH + postname, 'r', encoding = 'utf-8-sig').readlines()
        
    # if this post has only one line : ignore
    if len(line_list) == 1 and random.randint(0, 10) <= 9 :
        continue
        
    # get words from this post
    post_word_list = []
    for i, line in enumerate(line_list) :
        line = re.sub("= = = = = = >", "======>", line)
        line = re.sub("= = >", "==>", line)
        
        if W2V_BY_EACH_WORD : line = re.sub(r" +", " ", line).split() + ["\n"]
        else : line += "\n"
        
        w2v_train_list.append(line)
        total_word_count += len(line)
        post_word_list += line
        
        if len(post_word_list) > POST_LENGTH_MAX :
            break
        
    # if this post is too short : ignore
    if len(post_word_list) < POST_LENGTH_MIN and random.randint(0, 10) <= 9 :
        continue
    # put ending character at the end of a post
    if USE_ENDING_MARK :
        if W2V_BY_EACH_WORD :
            post_word_list.append(ENDING_MARK)
        else :
            post_word_list += ENDING_MARK
    rnn_train_input.append(post_word_list)  
random.shuffle(rnn_train_input)
# end for

if not CREATE_NEW_W2V :
    try :
        if W2V_BY_EACH_WORD : word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
        else : word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
    except :
        print("couldn't find wordvec model file. creating new model file...")
        CREATE_NEW_W2V = True

if CREATE_NEW_W2V :
    if not W2V_BY_EACH_WORD : 
        MIN_COUNT = 1
    word_model = word2vec.Word2Vec(w2v_train_list, iter = W2V_ITER, sg = 1, size = WV_SIZE, window = 6, workers = 4, min_count = MIN_COUNT)
    if W2V_BY_EACH_WORD : word_model.save("myword2vec_by_word.model")
    else :  word_model.save("myword2vec_by_char.model")

word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
del word_model

print("\ntotal_word_count: ", total_word_count)
print("\nvector size: ", WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
print(word_vector.most_similar("貓", topn = 10))
#for i in range(0, 10) : print(rnn_train_input[i])
### END PREPARE TRAINING DATA ###

def make_input_matrix(word_list) :
    input_matrix = np.zeros([1, len(word_list) + 1, WV_SIZE])
    i = 1 # begin at 1 because starting symbol is zero vector
    for word in word_list :
        try :
            input_matrix[0, i] = word_vector[word]
            i += 1
        except KeyError :
            for c in word :
                try :
                    input_matrix[0, i] = word_vector[word]
                    i += 1
                except KeyError :
                    continue

    return input_matrix
    
def make_label_matrix(word_list) :
    label_matrix = np.zeros([1, len(word_list) + 1, 1], dtype=np.int32)
    i = 0
    for word in word_list :
        try :
            label_matrix[0, i, 0] = word_vector.vocab[word].index # because sparse_categorical
            i += 1
        except KeyError :
            for c in word :
                try :
                    label_matrix[0, i, 0] = word_vector.vocab[word].index
                    i += 1
                except KeyError :
                    continue    
    return label_matrix

x_train_list = []
y_train_list = []
for post in rnn_train_input :
    x_train_list.append(make_input_matrix(post))
    y_train_list.append(make_label_matrix(post))

# make batch training data
def generate_sentences() :
    i = 0
    l = len(rnn_train_input)
    while 1:
        yield x_train_list[i], y_train_list[i]
        i += 1
        if i >= l : i = 0

def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))

### NETWORK MODEL ###
print("\nUSE_SAVED_MODEL: ", USE_SAVED_MODEL, "\nRNN_UNIT: ", RNN_UNIT, "\nOUTPUT_NUMBER:", OUTPUT_NUMBER)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = 0.1, momentum = 0.9, nesterov = True, decay = 1e-4)
    adam = optimizers.Adam(lr = 0.001)
    model = Sequential()
    model.add(LSTM(RNN_UNIT, input_shape = [None, WV_SIZE], return_sequences = True))
    model.add(Dense(VOCAB_SIZE, activation = "softmax"))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = sgd, metrics = ["sparse_categorical_accuracy"])
model.summary()
model.fit_generator(generator = generate_sentences(), steps_per_epoch = len(rnn_train_input), epochs = EPOCHS, verbose = 1)
model.save(SAVE_MODEL_NAME)

outfile = open("output.txt", "w+", encoding = "utf-8-sig")

def sample(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, prediction, 1)
    return probas

for out_i in range(OUTPUT_NUMBER) :
    output_sentence = [] # zero vector
    for n in range(120) :
        y_test = model.predict(make_input_matrix(output_sentence))
        # we only need y_test[0, y_test.shape[1] - 1] because it tells the next missing word
        y_test = sample(y_test[0, y_test.shape[1] - 1], temperature = 0.6)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        if next_word == ENDING_MARK : break
        if next_word == '\n' and output_sentence[-1] == '\n' : continue
        output_sentence.append(next_word)
    output_sentence.append("\n\n")
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write("{out_i}\n")
outfile.close()

