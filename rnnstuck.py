import os
import re
import sys
import numpy as np
import math
import h5py
import train_w2v as w2vparam
from gensim.models import word2vec
from keras import activations, optimizers
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras.layers import Activation, Concatenate, ConvLSTM2D, CuDNNLSTM, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LeakyReLU, LSTM, Masking, multiply, BatchNormalization, Permute, RepeatVector, Reshape, TimeDistributed

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# -1 : Use CPU; 0 or 1 : Use GPU

PAGE_LENGTH_MAX = None # set None to be unlimited
PAGE_LENGTH_MIN = 32
SENTENCE_LENGTH_MIN = 3

USE_SAVED_MODEL = False

MAX_TIMESTEP = 24 # set None to be unlimited
FIXED_TIMESTEP = False
if FIXED_TIMESTEP and MAX_TIMESTEP > PAGE_LENGTH_MIN :
    MAX_TIMESTEP = PAGE_LENGTH_MIN

ZERO_OFFSET = False # only valid when FIXED_TIMESTEP == False
USE_ATTENTION = True # only valid when FIXED_TIMESTEP == True

SAVE_MODEL_NAME = "rnnstuck_model.h5"

VOCAB_SIZE = -1
RNN_UNIT = [64] # 1 nvidia gt730 gpu: lstm(300) is limit
BATCH_SIZE = 196
EPOCHS = 32
VALIDATION_NUMBER = 100

OUTPUT_NUMBER = 4
OUTPUT_TIME_STEP = 128

# 4000 sample +
# W2V_BY_VOCAB = True +
# 100 unit lstm +
# 1 epoch
#    ==> ~20 minute

print("\nW2V_BY_VOCAB: ", w2vparam.W2V_BY_VOCAB, "\nPAGE_LENGTH_MAX", PAGE_LENGTH_MAX, "\nPAGE_LENGTH_MIN", PAGE_LENGTH_MIN)
if not MAX_TIMESTEP :
    if MAX_TIMESTEP > PAGE_LENGTH_MIN :
        print("Error: PAGE_LENGTH_MIN must bigger than MAX_TIMESTEP")
        exit() 

### PREPARE TRAINING DATA ###
page_list, total_word_count = w2vparam.get_train_data(page_length_min = PAGE_LENGTH_MIN, sentence_length_min = SENTENCE_LENGTH_MIN)
np.random.shuffle(page_list)

### LOAD WORD MODEL ###
word_model_name = "myword2vec_by_word.model" if w2vparam.W2V_BY_VOCAB else "myword2vec_by_char.model"
try :
    word_model = word2vec.Word2Vec.load(word_model_name)
except :
    print("couldn't find wordvec model file", word_model_name, "exiting program...")
    exit()
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
del word_model

print("total_word_count: ", total_word_count)
print("vector size: ", w2vparam.WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
print("\n貓:", word_vector.most_similar("貓", topn = 10))
#for i in range(0, 10) : print(page_list[i])

### PREPARE TRAINING DATA ###
def make_input_matrix(word_list, sentence_length_limit = None) :
    dim = w2vparam.WV_SIZE
    if sentence_length_limit :
        input_matrix = np.zeros([1, sentence_length_limit, dim])
    else :
        input_matrix = np.zeros([1, len(word_list), dim])
    
    if sentence_length_limit :
        word_list = word_list[ -sentence_length_limit : ] # only keep last few words if has sentence_length_limit

    for i, word in enumerate(word_list) :
        try :
            input_matrix[0, i] = word_vector[word] # add one because zero is masked
        except KeyError :
            for c in word :
                try :
                    input_matrix[0, i] = word_vector[word]
                except KeyError :
                    continue
    return input_matrix
    
def make_label_matrix(word_list) :
    label_matrix = np.zeros([1, len(word_list), 1], dtype=np.int32)
    word_list = word_list[1 : ] # delete start mark
    for i, word in enumerate(word_list) :
        try :
            label_matrix[0, i, 0] = word_vector.vocab[word].index # because sparse_categorical
        except KeyError :
            for c in word :
                try :
                    label_matrix[0, i, 0] = word_vector.vocab[word].index
                except KeyError :
                    continue
    # don't want last element in label_matrix be zero vecter, so make it to be ending mark
    label_matrix[0, -1, 0] = word_vector.vocab[w2vparam.ENDING_MARK].index
    return label_matrix

train_data_list = []
label_data_list = []
for page in page_list :
    train_data_list.append(make_input_matrix(page))
    label_data_list.append(make_label_matrix(page))

# make batch training data
def generate_sentences(max_time_step, batch_size, zero_offset = False, use_prevent_end_mark = True) :
    train_input_length = len(train_data_list)
    end_mark_factor = MAX_TIMESTEP
    #print("end_mark_factor:", end_mark_factor)
    print_counter = 0
    while 1:
        x = np.zeros((batch_size, max_time_step, w2vparam.WV_SIZE))
        y = np.zeros((batch_size, 1))
        i = 0
        while(i < batch_size) :
            n = np.random.randint(train_input_length)
            is_no_endmark = int(np.random.randint(end_mark_factor) / end_mark_factor + 1)
            # decide answer, this number is INCLUSIVE
            # and then, decide timestep length 
            if FIXED_TIMESTEP :
                answer = np.random.randint(max_time_step, train_data_list[n].shape[1]) - is_no_endmark
                time_step = max_time_step
            elif zero_offset :    
                answer = np.random.randint(0, min(train_data_list[n].shape[1] - is_no_endmark, max_time_step))
                time_step = answer + 1 # the length from 0 to answer is answer+1
            else :
                answer = np.random.randint(0, train_data_list[n].shape[1] - is_no_endmark)
                time_step = np.random.randint(1, min(answer + 2, max_time_step + 1))
            x[i, : time_step] = train_data_list[n][:, answer + 1 - time_step : answer + 1]
            # answer + 1 because need to include train_data_list[answer]
            y[i] = label_data_list[n][:, answer]
            i += 1
        # end while i < batch_size
        '''
        if print_counter < 1 :
            print(x.shape, y.shape)
            print(word_vector.wv.index2word[int(y[0, :])])
            print_counter += 1
        '''
        yield x, y

### CALLBACK FUNCTIONS ###
def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))
    
def lrate_epoch_decay(epoch) :
    init_lr = learning_rate
    decay = 0.707
    e = min(8, (epoch + 1) // 2) # 2 epoches per decay, with max e = 8
    return init_lr * math.pow(decay, e)
    
lr_scheduler = LearningRateScheduler(lrate_epoch_decay)
early_stop = EarlyStopping(monitor = "loss", min_delta = 0.001, patience = EPOCHS // 2)

def sample(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    return np.random.multinomial(1, prediction, 1)
    
def predict_output_sentence(predict_model, temperature, max_output_length, initial_input_sentence = None) :
    output_sentence = []
    if initial_input_sentence :
        output_sentence += initial_input_sentence
    for n in range(max_output_length) :
        input_array = make_input_matrix(output_sentence, sentence_length_limit = MAX_TIMESTEP)
        y_test = model.predict(input_array)
        y_test = sample(y_test[0], temperature)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]   
        output_sentence.append(next_word)
        if next_word == w2vparam.ENDING_MARK : break
    output_sentence.append("\n")
    return output_sentence

def output_to_file(filename, output_number, max_output_length) :
    outfile = open(filename, "w+", encoding = "utf-8-sig")
    for out_i in range(output_number) :
        output_sentence = predict_output_sentence(model, 0.75, max_output_length, np.random.choice(page_list)[0 : 2])
        output_string = ""
        for word in output_sentence :
            output_string += word
        outfile.write(output_string)
        outfile.write(">>>>>>>>\n")
    outfile.close()

class OutputPrediction(Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        output_to_file("output.txt", 4, OUTPUT_TIME_STEP)

pred_outputer = OutputPrediction()

### NETWORK MODEL ###
STEPS_PER_EPOCH = total_word_count // BATCH_SIZE // 2
learning_rate = 0.001

print("\nUSE_SAVED_MODEL:", USE_SAVED_MODEL)
print("max time step:", MAX_TIMESTEP, "\nrnn units:", RNN_UNIT, "\nbatch size:", BATCH_SIZE, "\nvalidation number:", VALIDATION_NUMBER, "\noutput number:", OUTPUT_NUMBER)
print("step per epoch:", STEPS_PER_EPOCH, "\nlearning_rate:", learning_rate)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = learning_rate, momentum = 0.9, nesterov = True, decay = 0.0)
    rmsprop = optimizers.RMSprop(lr = learning_rate, decay = 0.0)
    adam = optimizers.Adam(lr = learning_rate, decay = 0.0)
    
    ## make model
    input_layer = Input([MAX_TIMESTEP, w2vparam.WV_SIZE])
    if not FIXED_TIMESTEP :
        rnn_layer = Masking(mask_value = 0.)(input_layer)
    else :
        rnn_layer = input_layer
    
    for i, v in enumerate(RNN_UNIT) :
        rnn_layer = LSTM(v, return_sequences = (i != len(RNN_UNIT) - 1) or USE_ATTENTION, stateful = False)(rnn_layer)
        rnn_layer = BatchNormalization()(rnn_layer)
    if USE_ATTENTION :
        print(rnn_layer.shape)
        attention = Dense(1, activation = "softmax")(rnn_layer)
        print(attention.shape)
        postproc_layer = multiply([attention, rnn_layer])
        postproc_layer = Lambda(lambda x: K.sum(x, axis = 1))(postproc_layer)
        postproc_layer = BatchNormalization()(postproc_layer)
    else :
        postproc_layer = rnn_layer
    guess_next = Dense(VOCAB_SIZE, activation = "softmax")(postproc_layer)
    
    model = Model(input_layer, guess_next)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = rmsprop, metrics = ["sparse_categorical_accuracy"])
    
model.summary()

model.fit_generator(generator = generate_sentences(MAX_TIMESTEP, BATCH_SIZE, zero_offset = ZERO_OFFSET),
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = EPOCHS, 
                    verbose = 1,
                    callbacks = [lr_scheduler, early_stop, pred_outputer],
                    validation_data = generate_sentences(MAX_TIMESTEP, VALIDATION_NUMBER, zero_offset = True), 
                    validation_steps = 1)
model.save(SAVE_MODEL_NAME)


