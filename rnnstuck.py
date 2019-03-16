import os
import re
import sys
import numpy as np
import math
import h5py
import train_w2v_model as w2vparas
from gensim.models import word2vec
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.callbacks import Callback, LearningRateScheduler
from keras.layers import Activation, Dense, dot, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, Masking, multiply, BatchNormalization, Permute, RepeatVector, Reshape, TimeDistributed

#os.environ["CUDA_VISIBLE_DEVICES"] = '1' # -1 : Use CPU; 0 or 1 : Use GPU

PAGE_LENGTH_MAX = 256 # set None to be unlimited
PAGE_LENGTH_MIN = 16

USE_SAVED_MODEL = False
Use_FULL_SEQ = False # only relevant when FIXED_PAGE_LENGTH is given
USE_ATTENTION = False # only relevant when Use_FULL_SEQ is True
SAVE_MODEL_NAME = "rnnstuck_model.h5"

VOCAB_SIZE = -1
RNN_UNIT = 40 # 1 core nvidia gt730 gpu: lstm(300) is limit
BATCH_SIZE = 128
EPOCHS = 128
VALIDATION_NUMBER = 100
OUTPUT_NUMBER = 8
# 4000 sample +
# W2V_BY_VOCAB = True +
# 100 unit lstm +
# 1 epoch
#    ==> ~20 minute

print("\nW2V_BY_VOCAB: ", w2vparas.W2V_BY_VOCAB)

### PREPARE TRAINING DATA ###
total_word_count = 0
page_list = []
for count, pagename in enumerate(w2vparas.PAGENAME_LIST[w2vparas.SAMPLE_BEGIN : w2vparas.SAMPLE_END]) :
    if w2vparas.W2V_BY_VOCAB :
        line_list = open(w2vparas.CUT_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
    else :
        line_list = open(w2vparas.PROC_PATH + pagename, 'r', encoding = 'utf-8-sig').readlines()
        
    # get words from this page
    if w2vparas.W2V_BY_VOCAB :
        this_page_words = []
    else :
        this_page_words = ""
    for i, line in enumerate(line_list) :
    
        if PAGE_LENGTH_MAX :
            if len(this_page_words) + len(line) >= PAGE_LENGTH_MAX :
                break
        
        if re.match(r"[\s\.\-/]{4,}", line) : continue # ignore morse code
        if line == "\n" : continue 
        if w2vparas.W2V_BY_VOCAB : line = line.split() + ['\n']
        
        total_word_count += len(line)
        this_page_words += line
    
    # if this page is too short : ignore
    if len(this_page_words) < PAGE_LENGTH_MIN :
        continue
    # put ending character at the end of a page
    if w2vparas.USE_ENDING_MARK :
        if w2vparas.W2V_BY_VOCAB :
            this_page_words.append(w2vparas.ENDING_MARK_WORD)
        else :
            this_page_words += w2vparas.ENDING_MARK_CHAR
    page_list.append(this_page_words)  
    # end for
np.random.shuffle(page_list)
### END PREPARE TRAINING DATA ###

### LOAD WORD MODEL ###
word_model_name = "myword2vec_by_word.model" if w2vparas.W2V_BY_VOCAB else "myword2vec_by_char.model"
try :
    word_model = word2vec.Word2Vec.load(word_model_name)
except :
    print("couldn't find wordvec model file", word_model_name, "exiting program...")
    exit()
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
del word_model

print("total_word_count: ", total_word_count)
print("vector size: ", w2vparas.WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
print("\n貓:", word_vector.most_similar("貓", topn = 10))
#for i in range(0, 10) : print(page_list[i])

def make_input_matrix(word_list, sentence_length = None) :
    if sentence_length :
        input_matrix = np.zeros([1, sentence_length, w2vparas.WV_SIZE])
    else :
        input_matrix = np.zeros([1, len(word_list) + 1, w2vparas.WV_SIZE])
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
    
def make_label_matrix(word_list, sentence_length = None) :
    if sentence_length :
        label_matrix = np.zeros([1, sentence_length], dtype=np.int32)
    else :
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
def generate_sentences(max_time_step, batch_size = 1) :
    train_input_length = len(page_list)
    #print_counter = 0
    loop_counter = 0
    seq_length = 0
    while 1:
        post_nums = np.random.choice(train_input_length, batch_size)        
        x = np.zeros((batch_size, max_time_step, w2vparas.WV_SIZE))
        y = np.zeros((batch_size, 1)) 
    
        for i, n in enumerate(post_nums) :
            this_length = np.random.randint(0, min(max_time_step, x_train_list[n].shape[1]))
            x[i, : this_length + 1] = x_train_list[n][:, : this_length + 1]
            y[i] = y_train_list[n][:, this_length]

        #if print_counter < 1 :
        #    print(x.shape, y.shape)
        #    print(word_vector.wv.index2word[int(y[0, 0])])
        #    print_counter += 1
        yield x, y

### NETWORK MODEL ###
STEPS_PER_EPOCH = math.floor(total_word_count / BATCH_SIZE / 1.2)
learning_rate = 0.001

print("\nUSE_SAVED_MODEL:", USE_SAVED_MODEL, "\nrnn units:", RNN_UNIT, "\nbatch size:", BATCH_SIZE, "\nvalidation number:", VALIDATION_NUMBER, "\noutput number:", OUTPUT_NUMBER)
print("step per epoch:", STEPS_PER_EPOCH, "\nlearning_rate:", learning_rate)

def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))
    
def lrate_epoch_decay(epoch) :
    init_lr = learning_rate
    epoch_per_decay = 2
    decay = 0.5
    e = min(8, math.floor((epoch + 1) / epoch_per_decay))
    return init_lr * math.pow(decay, e)
    
lr_scheduler = LearningRateScheduler(lrate_epoch_decay)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = learning_rate, momentum = 0.9, nesterov = True, decay = 0.0)
    rmsprop = optimizers.RMSprop(lr = learning_rate, decay = 0.0)
    adam = optimizers.Adam(lr = learning_rate, decay = 0.0)
    
    ## make model
    input_layer = Input([PAGE_LENGTH_MAX, w2vparas.WV_SIZE])
    masked_layier = Masking(mask_value = 0.)(input_layer)
    if Use_FULL_SEQ :
        rnn_layer = LSTM(RNN_UNIT, return_sequences = True, stateful = False)(masked_layier)
        if USE_ATTENTION :
            attention = Permute((2, 1))(rnn_layer)
            attention = Dense(1, activation = "softmax")(attention)
            attention = Permute((2, 1))(attention)
            attention = RepeatVector(PAGE_LENGTH_MAX)(attention)
            x = multiply([attention, rnn_layer])
            x = Flatten()(x)
        else :
            x = Flatten()(rnn_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
    else :
       rnn_layer = LSTM(RNN_UNIT, return_sequences = False, stateful = False)(masked_layier)
       x = BatchNormalization()(rnn_layer)
       #x = Dropout(0.5)(x)
    guess_next = Dense(VOCAB_SIZE, activation = "softmax")(x)
    
    model = Model(input_layer, guess_next)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = rmsprop, metrics = ["sparse_categorical_accuracy"])
    
model.summary()

model.fit_generator(generator = generate_sentences(PAGE_LENGTH_MAX, batch_size = BATCH_SIZE),
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = EPOCHS, 
                    verbose = 1,
                    callbacks = [lr_scheduler],
                    validation_data = generate_sentences(PAGE_LENGTH_MAX, batch_size = VALIDATION_NUMBER), 
                    validation_steps = 1)
model.save(SAVE_MODEL_NAME)

def sample(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, prediction, 1)
    return probas
    
def predict_output_sentence(predict_model, initial_input_sentence = None) :
    output_sentence = [] # zero vector
    if initial_input_sentence :
        output_sentence += initial_input_sentence
    for n in range(250) :
        input_array = make_input_matrix(output_sentence, sentence_length = PAGE_LENGTH_MAX)
        y_test = model.predict(input_array)
        y_test = sample(y_test[0], 0.75)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        output_sentence.append(next_word)
        if next_word == w2vparas.ENDING_MARK_WORD or next_word == w2vparas.ENDING_MARK_CHAR : break
    output_sentence.append("\n")
    return output_sentence

outfile = open("output.txt", "w+", encoding = "utf-8-sig")
for out_i in range(OUTPUT_NUMBER) :
    output_sentence = predict_output_sentence(model, np.random.choice(page_list)[0])
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write(">>>>>>>>\n")
outfile.close()

