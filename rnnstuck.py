import os
import re
import numpy as np
import jieba_zhtw as jb
import random
import h5py
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, SimpleRNN, LSTM, GRU
from gensim.models import word2vec

#os.environ["CUDA_VISIBLE_DEVICES"] = '1' # -1 : Use CPU; 0 or 1 : Use GPU

PROCD_POSTS_DIR_PATH = "processed_posts/"
CUT_POSTS_DIR_PATH = "cut_posts/"

CREATE_NEW_JIEBA = False
CREATE_NEW_W2V = False
W2V_BY_EACH_WORD = False # False: Create w2v model by each character
USE_ENDING_CHARACTER = False
ENDING_CHARACTER = '\t'

W2V_ITER = 1
VOCAB_SIZE = -1

SAMPLE_BEGIN = 0
SAMPLE_END = 5420
POST_LENGTH_MAX = 8000
POST_LENGTH_MIN = 8

USE_SAVED_MODEL = False
WORD_VEC_SIZE = 200
RNN_UNIT = 100 # 1 core nvidia gt730 gpu: lstm(300) is limit
EPOCHS = 20
OUTPUT_NUMBER = 80
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
for filename in os.listdir(PROCD_POSTS_DIR_PATH) :
    POSTNAME_LIST.append(filename)
POSTNAME_LIST = sorted(POSTNAME_LIST, key = sorting_file_name)

### JIEBA ###
print("CREATE_NEW_JIEBA: ", CREATE_NEW_JIEBA)
if CREATE_NEW_JIEBA :
    jb.dt.cache_file = 'jieba.cache.zhtw'
    jb.load_userdict('hs_dict.txt')

    for postname in POSTNAME_LIST :
        jieba_in_string = open(PROCD_POSTS_DIR_PATH + postname, 'r', encoding = "utf-8-sig").read() 
        cut_post = jb.cut(jieba_in_string, cut_all = False)
        open(CUT_POSTS_DIR_PATH + postname, 'w+', encoding = 'utf-8-sig').write(" ".join(cut_post))    
### JIEBA ###

print("\nW2V_BY_EACH_WORD: ", W2V_BY_EACH_WORD, "\nCREATE_NEW_W2V: ", CREATE_NEW_W2V, "\niter: ", W2V_ITER)

### W2V ###
if CREATE_NEW_W2V :
    w2v_train_sentence_list = []
    print("\npreparing sentence to train...")
    for postname in POSTNAME_LIST :
        if W2V_BY_EACH_WORD :
            line_list = open(CUT_POSTS_DIR_PATH + postname, 'r', encoding = 'utf-8-sig').readlines()
        else :
            line_list = open(PROCD_POSTS_DIR_PATH + postname, 'r', encoding = 'utf-8-sig').readlines()
        for line in line_list :
            line = re.sub("= = = = = = >", "======>", line)
            line = re.sub("= = >", "==>", line)
            if W2V_BY_EACH_WORD :
                line_word_list = line.split() + ['\n']
                w2v_train_sentence_list.append(line_word_list)
            else :
                w2v_train_sentence_list.append(line)
        # put ending character at the end of a post's last sentence
        if USE_ENDING_CHARACTER :
            if W2V_BY_EACH_WORD :
                w2v_train_sentence_list[-1].append(ENDING_CHARACTER)
            else :
                w2v_train_sentence_list[-1] += ENDING_CHARACTER
    print("training w2v model...")
    
    if W2V_BY_EACH_WORD : MIN_COUNT = 5
    else : MIN_COUNT = 1
    word_model = word2vec.Word2Vec(w2v_train_sentence_list, iter = W2V_ITER, size = WORD_VEC_SIZE, window = 6, workers = 4, min_count = MIN_COUNT)
    
    if W2V_BY_EACH_WORD : word_model.save("myword2vec_by_word.model")
    else : word_model.save("myword2vec_by_char.model")
    del w2v_train_sentence_list
    
else :
    if W2V_BY_EACH_WORD :
        word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
    else :
        word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
        
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
print("\nvector size: ", WORD_VEC_SIZE, "\nvocab size: ", VOCAB_SIZE)
#print(word_vector.most_similar("貓", topn = 10))
### W2V ### 

### PREPARE TRAINING DATA ###
total_word_count = 0
rnn_train_input = []
for count, postname in enumerate(POSTNAME_LIST[SAMPLE_BEGIN : SAMPLE_END]) :
    if W2V_BY_EACH_WORD :
        line_list = open(CUT_POSTS_DIR_PATH + postname, 'r', encoding = 'utf-8-sig').readlines()
    else :
        line_list = open(PROCD_POSTS_DIR_PATH + postname, 'r', encoding = 'utf-8-sig').readlines()
    # if this post has only one line : ignore
    if len(line_list) == 1 and random.randint(0, 4) <= 3 :
        continue
    # get words from this post
    post_word_list = []
    for i, line in enumerate(line_list) :
        line = re.sub("= = = = = = >", "======>", line)
        line = re.sub("= = >", "==>", line)
        if W2V_BY_EACH_WORD :
            sentence = re.sub(r" +", " ", line).split()
            total_word_count += len(sentence)
            sentence.append('\n')
            post_word_list += sentence
        else :
            total_word_count += len(line)
            post_word_list += line
        if len(post_word_list) > POST_LENGTH_MAX :
            break
    # if this post is too short : ignore
    if len(post_word_list) < POST_LENGTH_MIN and random.randint(0, 4) <= 3 :
        continue
    # put ending character at the end of a post
    if W2V_BY_EACH_WORD : post_word_list += ENDING_CHARACTER
    rnn_train_input.append(post_word_list)
print("\ntotal_word_count: ", total_word_count)
#for i in range(0, 10) : print(rnn_train_input[i])
### PREPARE TRAINING DATA ###

def make_input_matrix(word_list) :
    input_matrix = np.zeros([1, len(word_list) + 1])
    i = 1 # begin at 1 because starting symbol is zero vector
    for word in word_list :
        try :
            input_matrix[0, i] = word_vector.vocab[word].index
        except KeyError :
            for c in word :
                try :
                    input_matrix[0, i] = word_vector.vocab[word].index
                except KeyError :
                    continue
        i += 1
    return input_matrix
    
def make_label_matrix(word_list) :
    label_matrix = np.zeros([1, len(word_list) + 1, 1], dtype=np.int32)
    i = 0
    for word in word_list :
        try :
            label_matrix[0, i, 0] = word_vector.vocab[word].index # because sparse_categorical
        except KeyError :
            for c in word :
                try :
                    label_matrix[0, i, 0] = word_vector.vocab[word].index
                except KeyError :
                    continue
        i += 1
    return label_matrix

# make batch training data
def generate_batch_snetences() :
    epoch_count = 0
    while 1:
        for post in rnn_train_input :
            x_train = make_input_matrix(post)
            y_train = make_label_matrix(post)
            #print('\nx.shape: ', x_train.shape, 'y.shape: ', y_train.shape)
            #print('y: ', y_train)
            yield x_train, y_train
        epoch_count += 1

 
### NETWORK MODEL ###
print("\nUSE_SAVED_MODEL: ", USE_SAVED_MODEL, "\nRNN_UNIT: ", RNN_UNIT, "\nOUTPUT_NUMBER:", OUTPUT_NUMBER)

if USE_SAVED_MODEL :
    model = load_model("rnn_stuck_model.h5")
else :
    optimizer = optimizers.RMSprop(lr = 0.001)
    model = Sequential()
    model.add(Embedding(input_dim = VOCAB_SIZE, output_dim = WORD_VEC_SIZE))
    model.add(GRU(RNN_UNIT, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Dense(VOCAB_SIZE, activation = "softmax"))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['sparse_categorical_accuracy'])
model.summary()
model.fit_generator(generator = generate_batch_snetences(), steps_per_epoch = len(rnn_train_input), epochs = EPOCHS, shuffle = True, verbose = 1)
model.save("rnn_stuck_model.h5")

outfile = open("output.txt", "w+", encoding = "utf-8-sig")

def add_noise(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, prediction, 1)
    return probas

for out_i in range(OUTPUT_NUMBER) :
    output_sentence = [rnn_train_input[random.randint(0, len(rnn_train_input))][0]]
    for n in range(60) :
        y_test = model.predict(make_input_matrix(output_sentence))
        # we only need y_test[0, y_test.shape[1] - 1] because it tells the next missing word
        y_test = add_noise(y_test[0, y_test.shape[1] - 1], temperature = 0.6)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        if (output_sentence[-1] == '\n' and next_word == '\n') : continue
        output_sentence.append(next_word)
    if out_i % 10 == 0 : print("i:", out_i)
    output_sentence.append('\n')
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
outfile.close()

