import os
import re
import sys
import numpy as np
import math
import h5py
from configure import *
from train_w2v import *

from gensim.models import word2vec
from keras import activations, optimizers
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras.layers import Activation, Concatenate, ConvLSTM2D, CuDNNLSTM, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, Masking, multiply, BatchNormalization, Permute, RepeatVector, Reshape, TimeDistributed

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# -1 : Use CPU; 0 or 1 : Use GPU

print("\nW2V_BY_VOCAB: ", W2V_BY_VOCAB, "\nPAGE_LENGTH_MAX", PAGE_LENGTH_MAX, "\nPAGE_LENGTH_MIN", PAGE_LENGTH_MIN)
if MAX_TIMESTEP :
    if MAX_TIMESTEP > PAGE_LENGTH_MIN :
        print("Error: PAGE_LENGTH_MIN must bigger than MAX_TIMESTEP")
        exit() 

### PREPARE TRAINING DATA ###
page_list, total_word_count = get_train_data(page_length_min = PAGE_LENGTH_MIN, page_length_max = PAGE_LENGTH_MAX, line_length_min = LINE_LENGTH_MIN, line_length_max = LINE_LENGTH_MAX)
np.random.shuffle(page_list)

### LOAD WORD MODEL ###
word_model_name = "myword2vec_by_word.model" if W2V_BY_VOCAB else "myword2vec_by_char.model"
try :
    word_model = word2vec.Word2Vec.load(word_model_name)
except :
    print("couldn't find wordvec model file", word_model_name, "exiting program...")
    exit()
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
del word_model

print("total_word_count: ", total_word_count)
print("vector size: ", WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
#print("\n貓:", word_vector.most_similar("貓", topn = 10))
#for i in range(0, 10) : print(page_list[i])

### PREPARE TRAINING DATA ###
def make_input_matrix(word_list, sentence_length_limit = None) :
    dim = WV_SIZE
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
    label_matrix[0, -1, 0] = word_vector.vocab[ENDING_MARK].index
    return label_matrix

train_data_list = []
label_data_list = []
for page in page_list :
    train_data_list.append(make_input_matrix(page))
    label_data_list.append(make_label_matrix(page))

# make batch training data
def generate_random_sentences(max_timestep, batch_size, zero_offset = False) :
    train_input_length = len(train_data_list)
    is_maxtimestep_none = (max_timestep == None)
    end_mark_factor = max_timestep if max_timestep else OUTPUT_TIME_STEP
    print("end_mark_factor:", end_mark_factor)
    print_counter = 0
    
    x = np.zeros((batch_size, max_timestep, WV_SIZE))
    y = np.zeros((batch_size, 1))
    
    while 1 :
        if is_maxtimestep_none : max_timestep = np.random.randint(end_mark_factor)
        i = 0
        while(i < batch_size) :
            n = np.random.randint(train_input_length)
            is_no_endmark = int((np.random.randint(end_mark_factor) + 1) / end_mark_factor)
            if is_maxtimestep_none : max_timestep = train_data_list[n].shape[1]
            # decide answer, this number is INCLUSIVE
            # and then, decide timestep length
            if FIXED_TIMESTEP :
                answer = np.random.randint(max_timestep, train_data_list[n].shape[1]) - is_no_endmark
                time_step = max_timestep
            elif zero_offset :
                answer = np.random.randint(0, min(train_data_list[n].shape[1] - is_no_endmark, max_timestep))
                time_step = answer + 1 # the length from 0 to answer is answer+1
            else :
                answer = np.random.randint(0, train_data_list[n].shape[1] - is_no_endmark)
                time_step = np.random.randint(1, min(answer + 2, max_timestep + 1))
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
    # end while
# end def generate_random_sentences
        
def generate_post_as_sentence(max_timestep, batch_size, zero_offset = False) :
    train_input_length = len(train_data_list)
    n = 0
    #print(train_data_list[0].shape, label_data_list[0].shape)
    while 1 :
        yield train_data_list[n], label_data_list[n]
        n = (n + 1) % train_input_length

### CALLBACK FUNCTIONS ###
def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))
    
def lrate_epoch_decay(epoch) :
    init_lr = LEARNING_RATE
    e = min(LR_DECAY_POW_MAX, (epoch + 1) // LR_DECAY_INTV) # 2 epoches per decay, with max e = 8
    return init_lr * math.pow(LR_DECAY, e)
    
lr_scheduler = LearningRateScheduler(lrate_epoch_decay)
early_stop = EarlyStopping(monitor = "loss", min_delta = EARLYSTOP_MIN_DELTA, patience = EARLYSTOP_PATIENCE)

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
        y_test = predict_model.predict(input_array)
        y_test = sample(y_test[0], temperature)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]   
        output_sentence.append(next_word)
        if next_word == ENDING_MARK : break
    output_sentence.append("\n")
    return output_sentence

def output_to_file(filename, output_number, max_output_length) :
    outfile = open(filename, "w+", encoding = "utf-8-sig")
    predict_model = model if MAX_TIMESTEP else model_pred_last
    for out_i in range(output_number) :
        output_sentence = predict_output_sentence(predict_model, 0.75, max_output_length, np.random.choice(page_list)[0 : 2])
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
if MAX_TIMESTEP :
    STEPS_PER_EPOCH = int((total_word_count // BATCH_SIZE) * STEP_EPOCH_RATE)
else :
    STEPS_PER_EPOCH = int(len(page_list) * STEP_EPOCH_RATE)

print("\nUSE_SAVED_MODEL:", USE_SAVED_MODEL)
print("max time step:", MAX_TIMESTEP, "\nrnn units:", RNN_UNIT, "\nbatch size:", BATCH_SIZE, "\nvalidation number:", VALIDATION_NUMBER, "\noutput number:", OUTPUT_NUMBER)
print("step per epoch:", STEPS_PER_EPOCH, "\nlearning_rate:", LEARNING_RATE)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = LEARNING_RATE, momentum = 0.9, nesterov = True, decay = 0.0)
    rmsprop = optimizers.RMSprop(lr = LEARNING_RATE, decay = 0.0)
    adam = optimizers.Adam(lr = LEARNING_RATE, decay = 0.0)
    
    ## make model
    input_layer = Input([MAX_TIMESTEP, WV_SIZE])
    if not FIXED_TIMESTEP and MAX_TIMESTEP :
        rnn_layer = Masking(mask_value = 0.)(input_layer)
    else :
        rnn_layer = input_layer
    
    for i, v in enumerate(RNN_UNIT) :
        is_return_seq = (i != len(RNN_UNIT) - 1) or USE_ATTENTION or MAX_TIMESTEP == None
        if MAX_TIMESTEP == None :
            rnn_layer = CuDNNLSTM(v, return_sequences = is_return_seq, stateful = False)(rnn_layer)
        else :
            rnn_layer = LSTM(v, return_sequences = is_return_seq, stateful = False)(rnn_layer)
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
    
    if MAX_TIMESTEP == None :
        guess_last = Lambda(lambda x: x[:, -1])(guess_next)
        model_pred_last = Model(input_layer, guess_last)
 
model.summary()

generate_sentences = generate_random_sentences if MAX_TIMESTEP else generate_post_as_sentence

model.fit_generator(generator = generate_sentences(MAX_TIMESTEP, BATCH_SIZE, zero_offset = ZERO_OFFSET),
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = EPOCHS, 
                    verbose = 1,
                    callbacks = [lr_scheduler, early_stop, pred_outputer],
                    validation_data = generate_sentences(MAX_TIMESTEP, VALIDATION_NUMBER, zero_offset = True), 
                    validation_steps = 1)
if MAX_TIMESTEP :
    model.save(SAVE_MODEL_NAME)
else :
    model_pred_last.save(SAVE_MODEL_NAME)


