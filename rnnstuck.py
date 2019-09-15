import os
import re
import sys
import numpy as np
import math
import random
import h5py
from configure import *
from train_w2v import *
from gensim.models import word2vec
from keras import activations, optimizers
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping
from keras.layers import Activation, Bidirectional, Concatenate, ConvLSTM2D, CuDNNLSTM, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, Masking, multiply, BatchNormalization, Permute, RepeatVector, Reshape, TimeDistributed

# -1 : Use CPU; 0 or 1 : Use GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print("\nW2V_BY_VOCAB: ", W2V_BY_VOCAB, "\nPAGE_LENGTH_MAX", PAGE_LENGTH_MAX, "\nPAGE_LENGTH_MIN", PAGE_LENGTH_MIN)
if MAX_TIMESTEP :
    if MAX_TIMESTEP > PAGE_LENGTH_MIN :
        print("Warning: PAGE_LENGTH_MIN is smaller than MAX_TIMESTEP")

### PREPARE TRAINING DATA AND WORD MODEL ###
_p, _c = get_train_data()
word_model = make_new_w2v(_p, show_result = True)
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
del word_model

page_list, train_word_count = get_train_data(page_length_min = PAGE_LENGTH_MIN, page_length_max = PAGE_LENGTH_MAX, line_length_min = LINE_LENGTH_MIN, line_length_max = LINE_LENGTH_MAX)
random.shuffle(page_list)
print("total word count:", _c)
print("train word count:", train_word_count)
print("vector size: ", WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
#for i in range(0, 10) : print(page_list[i])

### PREPARE TRAINING DATA ###
def make_input_matrix(word_list, sentence_length_limit = None) :
    if USE_ENDING_MARK :
        word_list = word_list[ : -1]
    if sentence_length_limit :
        # only keep last few words if has sentence_length_limit
        word_list = word_list[ -sentence_length_limit : ]
        input_matrix = np.zeros([1, sentence_length_limit, WV_SIZE])
    else :
        input_matrix = np.zeros([1, len(word_list), WV_SIZE])
    
    i = 0 if USE_START_MARK else 1
    for word in word_list :
        try :
            input_matrix[0, i] = word_vector[word] # add one because zero is masked
        except KeyError :
            for c in word :
                try :
                    input_matrix[0, i] = word_vector[c]
                except KeyError :
                    continue # ignore
                i += 1
                if i >= len(word_list) : break
        i += 1
        if i >= len(word_list) : break
    return input_matrix
    
def make_label_matrix(word_list) :
    if USE_START_MARK :
        word_list = word_list[1 : ] # delete start mark
    label_matrix = np.zeros([1, len(word_list), 1], dtype=np.int32)
    
    i = 0
    for word in word_list :
        try :
            label_matrix[0, i, 0] = word_vector.vocab[word].index # because sparse_categorical
        except KeyError :
            for c in word :
                try :
                    label_matrix[0, i, 0] = word_vector.vocab[c].index
                except KeyError :
                    continue
                i += 1
                if i >= len(word_list) : break
        i += 1
        if i >= len(word_list) : break
    # don't want last element in label_matrix be zero vecter, so make it to be ending mark
    label_matrix[0, -1, 0] = word_vector.vocab[ENDING_MARK].index
    return label_matrix

train_data_list = []
label_data_list = []
for page in page_list :
    train_data_list.append(make_input_matrix(page))
    label_data_list.append(make_label_matrix(page))

def generate_train_data(max_timestep, batch_size, zero_offset) :
    train_list_len = len(train_data_list)
    while 1 :
        batch_num = random.sample(range(0, train_list_len), batch_size)
        if max_timestep :
            timestep_size = max_timestep
        elif USE_SEQ_LABEL or not zero_offset :
            max_length = min([train_data_list[b].shape[1] for b in batch_num])
            timestep_size = random.randint(1, max_length)
        else :
            max_length = max([train_data_list[b].shape[1] for b in batch_num])
            timestep_size = random.randint(1, max_length)

        x = np.zeros((batch_size, timestep_size, WV_SIZE))
        if USE_SEQ_LABEL :
            y = np.zeros((batch_size, timestep_size, 1), dtype = int)
        else :
            y = np.zeros((batch_size, 1), dtype = int)

        for i, b in enumerate(batch_num) :
            this_data_length = train_data_list[b].shape[1]
            if USE_SEQ_LABEL :
                timestep = min(this_data_length, timestep_size)
            else :
                timestep = random.randint(1, min(this_data_length, timestep_size))
            # 'answer' indocate a index of data in the label list (count from 0)
            # a train-label[x] is the one-hot rep of train-data[x+1]
            # so, if answer == 5, the longest train data we can get is [0 : 6], which is 6 in length
            # it means for a timestep, the smallest answer is timestep - 1
            if zero_offset :
                answer = timestep - 1
            else :
                answer = random.randint(timestep, this_data_length) - 1
            #try :
            x[i, : timestep] = train_data_list[b][:, answer - (timestep - 1) : answer + 1]
            if USE_SEQ_LABEL :
                y[i, : timestep] = label_data_list[b][:, answer - (timestep - 1) : answer + 1]
            else :
                y[i] = label_data_list[b][:, answer]
            #except :
            #    print("Index Error:", (this_data_length, answer, timestep))
        yield x, y

### CALLBACK FUNCTIONS ###
def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))

def sequential_sparse_categorical_crossentropy(y_true, y_pred) :
    # the shape of y_true is (samples, timesteps, 1)
    return K.sparse_categorical_crossentropy(y_true, y_pred, axis = 2)
    
def sequential_sparse_categorical_accuracy(y_true, y_pred) :
    # the shape of y_true is (samples, timesteps, 1)
    return K.mean(K.cast(K.equal(K.flatten(y_true),
                                 K.flatten(K.cast(K.argmax(y_pred, axis = -1), K.floatx())))
                , K.floatx()))
    
def lrate_epoch_decay(epoch) :
    init_lr = LEARNING_RATE
    e = min(LR_DECAY_POW_MAX, (epoch + 1) // LR_DECAY_INTV) # INTV epoches per decay, with max e
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
    if initial_input_sentence :
        output_sentence = initial_input_sentence
    elif W2V_BY_VOCAB :
        output_sentence = []
    else :
        output_sentence = ""
    for n in range(max_output_length) :
        input_array = make_input_matrix(output_sentence, sentence_length_limit = MAX_TIMESTEP)
        y_test = predict_model.predict(input_array)
        y_test = sample(y_test[0], temperature)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]   
        output_sentence += next_word
        if next_word == ENDING_MARK : break
    output_sentence += "\n"
    return output_sentence

def output_to_file(filename, output_number, max_output_length) :
    outfile = open(filename, "w+", encoding = "utf-8-sig")
    for _ in range(output_number) :
        seed = random.choice(page_list)[0 : 2]
        output_sentence = predict_output_sentence(model, 0.9, max_output_length, seed)
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
STEPS_PER_EPOCH = int((train_word_count - len(page_list) * 2) // BATCH_SIZE * STEP_EPOCH_RATE)

print("\nUSE_SAVED_MODEL:", USE_SAVED_MODEL)
print("max time step:", MAX_TIMESTEP, "\nuse zero offest:", ZERO_OFFSET, "\nuse seq label:", USE_SEQ_LABEL, "\nrnn units:", RNN_UNIT, "\nbatch size:", BATCH_SIZE, "\nvalidation number:", VALIDATION_NUMBER, "\noutput number:", OUTPUT_NUMBER)
print("step per epoch:", STEPS_PER_EPOCH, "\nlearning_rate:", LEARNING_RATE)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = LEARNING_RATE, momentum = 0.9, nesterov = True, decay = 0.0)
    rmsprop = optimizers.RMSprop(lr = LEARNING_RATE, decay = 0.0)
    adam = optimizers.Adam(lr = LEARNING_RATE, decay = 0.0)
    
    ## make model
    input_layer = Input([MAX_TIMESTEP, WV_SIZE])
    if MAX_TIMESTEP :
        rnn_layer = Masking(mask_value = 0.)(input_layer)
    else :
        rnn_layer = input_layer
    
    for i, v in enumerate(RNN_UNIT) :
        is_return_seq = (i != len(RNN_UNIT) - 1) or USE_ATTENTION or USE_SEQ_LABEL
        if MAX_TIMESTEP :
            rnn_layer = Bidirectional(LSTM(v, return_sequences = is_return_seq))(rnn_layer)
        else :
            rnn_layer = Bidirectional(CuDNNLSTM(v, return_sequences = is_return_seq))(rnn_layer)
        rnn_layer = Dropout(0.2)(rnn_layer)
    if USE_ATTENTION :
        attention = Dense(1, activation = "softmax")(rnn_layer)
        print("attention:", rnn_layer.shape, "to", attention.shape)
        postproc_layer = multiply([attention, rnn_layer])
        if not USE_SEQ_LABEL :
            postproc_layer = Lambda(lambda x: K.sum(x, axis = 1))(postproc_layer)
        postproc_layer = Dropout(0.2)(postproc_layer)
    else :
        postproc_layer = rnn_layer
    guess = Dense(VOCAB_SIZE, activation = "softmax")(postproc_layer)
    if USE_SEQ_LABEL :
        guess_last = Lambda(lambda x: x[:, -1])(guess)
        model = Model(input_layer, guess_last)
        model_train = Model(input_layer, guess)
        model_train.compile(loss = sequential_sparse_categorical_crossentropy, optimizer = adam, metrics = [sequential_sparse_categorical_accuracy])
    else :
        model = Model(input_layer, guess)
        model_train = model
        model_train.compile(loss = "sparse_categorical_crossentropy", optimizer = adam, metrics = ["sparse_categorical_accuracy"])

model_train.summary()
model_train.fit_generator(generator = generate_train_data(MAX_TIMESTEP, BATCH_SIZE, ZERO_OFFSET),
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = EPOCHS, 
                    verbose = 1,
                    callbacks = [lr_scheduler, early_stop, pred_outputer],
                    validation_data = generate_train_data(MAX_TIMESTEP, VALIDATION_NUMBER, True), 
                    validation_steps = 1)
model.save(SAVE_MODEL_NAME)
