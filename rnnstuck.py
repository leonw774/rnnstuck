import os
import re
import sys
import numpy as np
import random
import h5py
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, multiply, Permute, RepeatVector, Reshape, TimeDistributed
from gensim.models import word2vec

#os.environ["CUDA_VISIBLE_DEVICES"] = '1' # -1 : Use CPU; 0 or 1 : Use GPU

PROC_PATH = "processed_posts/"
CUT_PATH = "cut_posts/"

W2V_BY_EACH_WORD = True # False: Create w2v model by each character
USE_ENDING_MARK = True
ENDING_MARK_WORD = "<e>"
ENDING_MARK_CHAR = '\0'
CREATE_NEW_W2V = False

MIN_COUNT = 4
W2V_ITER = 6
VOCAB_SIZE = -1

SAMPLE_BEGIN = 0
SAMPLE_END = 5420
POST_LENGTH_MAX = 3000
POST_LENGTH_MIN = 3

USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "rnnstuck_model.h5"
WV_SIZE = 200
RNN_UNIT = 48 # 1 core nvidia gt730 gpu: lstm(300) is limit
BATCH_SIZE = 16
EPOCHS = 64 # (POST_LENGTH_MAX + POST_LENGTH_MIN) // 2
OUTPUT_NUMBER = 20
# 4000 sample +
# W2V_BY_EACH_WORD = True +
# 100 unit lstm +
# 1 epoch
#    ==> ~20 minute


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
        
    # get words from this post
    post_word_list = []
    for i, line in enumerate(line_list) :
        line = re.sub("= = = = = = >", "======>", line)
        line = re.sub("= = >", "==>", line)
        line = re.sub(r" +", " ", line)
    
        if line == "\n" : continue 
        if W2V_BY_EACH_WORD : line = line.split() + ['\n']
        
        w2v_train_list.append(line)
        total_word_count += len(line)
        post_word_list += line
        
        if len(post_word_list) > POST_LENGTH_MAX :
            break
    
    # if this post is too short : ignore
    if len(post_word_list) < POST_LENGTH_MIN and random.randint(0, 4) > 0 :
        continue    
    # put ending character at the end of a post
    if USE_ENDING_MARK :
        if W2V_BY_EACH_WORD :
            post_word_list.append(ENDING_MARK_WORD)
        else :
            post_word_list += ENDING_MARK_CHAR
    rnn_train_input.append(post_word_list)  
random.shuffle(rnn_train_input)
# end for

if not CREATE_NEW_W2V :
    try :
        if W2V_BY_EACH_WORD :
            word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
        else :
            word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
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
def generate_sentences(batch_size = 1) :
    train_input_length = len(rnn_train_input)
    while 1:
        post_nums = random.sample(range(0, train_input_length), batch_size)
        seq_length = POST_LENGTH_MAX
        for n in post_nums :
            seq_length = min(seq_length, x_train_list[n].shape[1])
        x = np.zeros((batch_size, seq_length, WV_SIZE))
        y = np.zeros((batch_size, seq_length, 1))
        for i, n in enumerate(post_nums) :
            x[i] = x_train_list[n][:, : seq_length] # index 0 to seq_length -1 
            y[i] = y_train_list[n][:, : seq_length]
        #print(x.shape, y.shape) 
        yield x, y

def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))

### NETWORK MODEL ###
print("\nUSE_SAVED_MODEL: ", USE_SAVED_MODEL, "\nRNN_UNIT: ", RNN_UNIT, "\nbatch size:", BATCH_SIZE, "\nOUTPUT_NUMBER:", OUTPUT_NUMBER)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = 0.01, momentum = 0.5, nesterov = True, decay = 1e-2)
    rmsprop = optimizers.RMSprop(lr = 0.001, decay = 2e-2)
    adam = optimizers.Adam(lr = 0.001, decay = 1e-2)
    
    ## make model
    model = Sequential()
    model.add(LSTM(RNN_UNIT, input_shape=(None, WV_SIZE), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Dense(VOCAB_SIZE, activation = "softmax"))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = rmsprop, metrics = ["sparse_categorical_accuracy"])
    
model.summary()

model.fit_generator(generator = generate_sentences(batch_size = BATCH_SIZE), steps_per_epoch = len(rnn_train_input), epochs = EPOCHS, verbose = 1 validation_split)
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
    output_sentence += random.choice(rnn_train_input)[0]
    for n in range(120) :
        y_test = model.predict(make_input_matrix(output_sentence))
        y_test = sample(y_test[0, -1], 0.75)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        if next_word == ENDING_MARK_WORD or next_word == ENDING_MARK_CHAR : break
        if next_word == '\n' :
            if len(output_sentence) == 0 or output_sentence[-1] == '\n' :
                n += 1
                continue
                
        output_sentence.append(next_word)
    output_sentence.append("\n\n")
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write("{out_i}\n")
outfile.close()

