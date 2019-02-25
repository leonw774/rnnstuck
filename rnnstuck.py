import os
import re
import sys
import numpy as np
import random
import h5py
import process_w2v_model as w2v_paras
from gensim.models import word2vec
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, multiply, Permute, RepeatVector, Reshape, TimeDistributed

#os.environ["CUDA_VISIBLE_DEVICES"] = '1' # -1 : Use CPU; 0 or 1 : Use GPU

PAGE_LENGTH_MAX = 2048
PAGE_LENGTH_MIN = 4
PAGE_LENGTH = 16 # set None to be varifyed length

USE_SAVED_MODEL = False
SAVE_MODEL_NAME = "rnnstuck_model.h5"
VOCAB_SIZE = -1
RNN_UNIT = 32 # 1 core nvidia gt730 gpu: lstm(300) is limit
BATCH_SIZE = 8
EPOCHS = 8
VALIDATION_NUMBER = 100
OUTPUT_NUMBER = 10
# 4000 sample +
# W2V_BY_VOCAB = True +
# 100 unit lstm +
# 1 epoch
#    ==> ~20 minute

def sorting_file_name(element) :
    if not element.endswith(".txt") :
        print(element, " <-- This is not a txt file. wtf did you mess out?")
        sys.exit(0)
    return int(element[0 : len(element) - 4])

print("\nprocessing path names...\n")
PAGENAME_LIST = []
for filename in os.listdir(w2v_paras.PROC_PATH) :
    PAGENAME_LIST.append(filename)
PAGENAME_LIST = sorted(PAGENAME_LIST, key = sorting_file_name)

print("W2V_BY_VOCAB: ", w2v_paras.W2V_BY_VOCAB, "\niters: ", w2v_paras.W2V_ITER)

### PREPARE TRAINING DATA ###
total_word_count = 0
page_list = []
for count, pagename in enumerate(PAGENAME_LIST[w2v_paras.SAMPLE_BEGIN : w2v_paras.SAMPLE_END]) :
    if w2v_paras.W2V_BY_VOCAB :
        line_list = open(w2v_paras.CUT_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
    else :
        line_list = open(w2v_paras.PROC_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
        
    # get words from this page
    this_page_word_list = []
    for i, line in enumerate(line_list) :
    
        if line == "\n" : continue 
        if w2v_paras.W2V_BY_VOCAB : line = line.split() + ['\n']
        
        total_word_count += len(line)
        this_page_word_list += line
        
        if len(this_page_word_list) > w2v_paras.PAGE_LENGTH_MAX :
            break
    
    # if this page is too short : ignore
    if len(this_page_word_list) < w2v_paras.PAGE_LENGTH_MIN :
        continue    
    # put ending character at the end of a page
    if w2v_paras.USE_ENDING_MARK :
        if w2v_paras.W2V_BY_VOCAB :
            this_page_word_list.append(w2v_paras.ENDING_MARK_WORD)
        else :
            this_page_word_list += w2v_paras.ENDING_MARK_CHAR
    page_list.append(this_page_word_list)  
    # end for
random.shuffle(page_list)
### END PREPARE TRAINING DATA ###

### LOAD WORD MODEL ###
word_model_name = "myword2vec_by_word.model" if w2v_paras.W2V_BY_VOCAB else "myword2vec_by_char.model"
try :
    word_model = word2vec.Word2Vec.load(word_model_name)
except :
    print("couldn't find wordvec model file", word_model_name, "exiting program...")
    exit()
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
STEPS_PER_EPOCH = total_word_count / BATCH_SIZE
if PAGE_LENGTH :
    STEPS_PER_EPOCH /= PAGE_LENGTH
del word_model

print("\ntotal_word_count: ", total_word_count)
print("\nvector size: ", w2v_paras.WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
print(word_vector.most_similar("貓", topn = 10))
for i in range(0, 10) : print(page_list[i])


def make_input_matrix(word_list) :
    input_matrix = np.zeros([1, len(word_list) + 1, w2v_paras.WV_SIZE])
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
for page in page_list :
    x_train_list.append(make_input_matrix(page))
    y_train_list.append(make_label_matrix(page))

# make batch training data
def generate_sentences(batch_size = 1, sentence_length = None) :
    train_input_length = len(page_list)
    print_counter = 0
    while 1:
        post_nums = random.sample(range(0, train_input_length), batch_size)
        # various length
        if sentence_length == None :
            seq_length = PAGE_LENGTH_MAX
            for n in post_nums :
                seq_length = min(seq_length, x_train_list[n].shape[1])
                
            x = np.zeros((batch_size, seq_length, w2v_paras.WV_SIZE))
            y = np.zeros((batch_size, seq_length, 1))
        
            for i, n in enumerate(post_nums) :
                offset = random.randint(0, x_train_list[n].shape[1] - seq_length)
                x[i] = x_train_list[n][:, offset : seq_length + offset] # index offset to seq_length + offset - 1
                y[i] = y_train_list[n][:, offset : seq_length + offset]
        # unified length
        else :
            seq_length = sentence_length
            
            x = np.zeros((batch_size, seq_length, w2v_paras.WV_SIZE))
            y = np.zeros((batch_size, 1))
            
            for i, n in enumerate(post_nums) :
                if seq_length <= x_train_list[n].shape[1] :
                        offset = random.randint(0, x_train_list[n].shape[1] - seq_length)
                        x[i] = x_train_list[n][:, offset : seq_length + offset] # index offset to seq_length + offset - 1 
                        y[i] = y_train_list[n][:, seq_length + offset - 1]
                else :
                    this_length = x_train_list[n].shape[1]
                    x[i, : this_length] = x_train_list[n][:, :] # index 0 to this_length - 1 
                    y[i] = y_train_list[n][:, this_length - 1]
        if print_counter < 0 :
            print(x.shape, y.shape)
            print_counter += 1
        yield x, y

def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))

### NETWORK MODEL ###
print("\nUSE_SAVED_MODEL: ", USE_SAVED_MODEL, "\nRNN_UNIT: ", RNN_UNIT, "\nbatch size:", BATCH_SIZE, "\nOUTPUT_NUMBER:", OUTPUT_NUMBER)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = 0.05, momentum = 0.5, nesterov = True, decay = (1/STEPS_PER_EPOCH))
    rmsprop = optimizers.RMSprop(lr = 0.005, decay = (1/STEPS_PER_EPOCH))
    adam = optimizers.Adam(lr = 0.001, decay = (1/STEPS_PER_EPOCH))
    
    ## make model
    model = Sequential()
    model.add(GRU(RNN_UNIT, input_shape = (PAGE_LENGTH, w2v_paras.WV_SIZE), return_sequences = (PAGE_LENGTH == None)))
    model.add(Dropout(0.2))
    model.add(Dense(VOCAB_SIZE, activation = "softmax"))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = rmsprop, metrics = ["sparse_categorical_accuracy"])
    
model.summary()

model.fit_generator(generator = generate_sentences(batch_size = BATCH_SIZE, sentence_length = PAGE_LENGTH),
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = EPOCHS, 
                    verbose = 1,
                    validation_data = generate_sentences(batch_size = VALIDATION_NUMBER, sentence_length = PAGE_LENGTH), 
                    validation_steps = 1)
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
    output_sentence += random.choice(page_list)[0]
    for n in range(100) :
        input_array = make_input_matrix(output_sentence)
        if PAGE_LENGTH :
            if input_array.shape[1] > PAGE_LENGTH :
                adjust_input_array = input_array[:, -PAGE_LENGTH :]
            else :
                adjust_input_array = np.zeros([1, PAGE_LENGTH, w2v_paras.WV_SIZE])
                adjust_input_array[:, : input_array.shape[1]] = input_array
            y_test = model.predict(adjust_input_array)
            y_test = sample(y_test[0], 0.7)
        else :
            y_test = model.predict(input_array)
            y_test = sample(y_test[0, -1], 0.7)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        output_sentence.append(next_word)
        if next_word == w2v_paras.ENDING_MARK_WORD or next_word == w2v_paras.ENDING_MARK_CHAR : break
    output_sentence.append("\n\n")
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write(">>>>>>>>\n")
outfile.close()

