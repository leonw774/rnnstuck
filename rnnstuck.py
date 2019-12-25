import os
import re
import sys
import numpy as np
import math
import random
import h5py
from configure import *
from train_w2v import *
from generate import *
from gensim.models import word2vec
from keras import activations, optimizers
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Bidirectional, Concatenate, CuDNNLSTM, Dense, Dropout, Embedding, Flatten, GRU, Input, Lambda, LSTM, Masking, Multiply, BatchNormalization, Permute, RepeatVector, Reshape, TimeDistributed

# -1 : Use CPU; 0 or 1 : Use GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print("\nW2V_BY_VOCAB:", W2V_BY_VOCAB, "\nWORD_LENGTH_MAX", WORD_LENGTH_MAX, "\nWORD_LENGTH_MIN", WORD_LENGTH_MIN, "\nLINE_LENGTH_MAX", LINE_LENGTH_MAX, "\nLINE_LENGTH_MIN", LINE_LENGTH_MIN)
if MAX_TIMESTEP :
    if MAX_TIMESTEP > WORD_LENGTH_MIN : print("Warning: WORD_LENGTH_MIN is smaller than MAX_TIMESTEP")

### PREPARE TRAINING DATA AND WORD MODEL ###
_p, _c = get_train_data()
word_model = make_new_w2v(_p, show_result = True)
word_vectors = word_model.wv
VOCAB_SIZE = word_vectors.syn0.shape[0]
del word_model

page_list, train_word_count = get_train_data(word_min = WORD_LENGTH_MIN, word_max = WORD_LENGTH_MAX, line_min = LINE_LENGTH_MIN, line_max = LINE_LENGTH_MAX)
random.shuffle(page_list)
print("total word count:", _c)
print("train word count:", train_word_count)
print("vector size: ", WV_SIZE, "\nvocab size: ", VOCAB_SIZE)
#for i in range(0, 10) : print(page_list[i])

### PREPARE TRAINING DATA ###
def make_training_matrices(word_list) :
    
    input_matrix = np.zeros([1, len(word_list), word_vectors.syn0.shape[1]])
    label_matrix = np.zeros([1, len(word_list), 1], dtype=np.int32)
    
    in_counter = 0 if USE_START_MARK else 1
    label_counter = 0

    for word in word_list :
        if word in word_vectors :
            if word != ENDING_MARK and in_counter < len(word_list) :
                input_matrix[0, in_counter] = word_vectors[word] # add one because zero is masked
                in_counter += 1
            if word != START_MARK :
                label_matrix[0, label_counter, 0] = word_vectors.vocab[word].index # because sparse_categorical
                label_counter += 1
        else :
            for c in word :
                if c in word_vectors :
                    if in_counter < len(word_list) :
                        input_matrix[0, in_counter] = word_vectors[c]
                        in_counter += 1
                    label_matrix[0, label_counter, 0] = word_vectors.vocab[c].index
                    label_counter += 1
                if label_counter >= len(word_list) :
                    break
        if label_counter >= len(word_list):
            break
    return input_matrix, label_matrix

train_data_list = []
label_data_list = []
for page in page_list :
    t, l = make_training_matrices(page)
    train_data_list.append(t)
    label_data_list.append(l)

train_test_split = len(page_list) * 11 // 12
test_data_list = train_data_list[train_test_split : ]
test_label_data_list = label_data_list[train_test_split : ]
train_data_list = train_data_list[ : train_test_split]
label_data_list = label_data_list[ : train_test_split]

def generate_batch(x, y, max_timestep, batch_size, zero_offset) :
    while 1 :
        batch_num = random.sample(range(0, len(x)), batch_size)
        if max_timestep :
            timestep_size = max_timestep
        else :
            max_length = max([x[b].shape[1] for b in batch_num])
            timestep_size = random.randint(1, max_length)

        bx = np.zeros((batch_size, timestep_size, WV_SIZE))
        by = np.zeros((batch_size, 1), dtype = int)

        for i, b in enumerate(batch_num) :
            this_data_length = x[b].shape[1]
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
            # bx from -timestep to end because the last timestep cannot be zero vector
            # x[b]'s last == answer + 1 because it need to include [answer]
            bx[i, : timestep] = x[b][:, answer - (timestep - 1) : answer + 1]
            by[i] = y[b][:, answer]
            #except :
            #    print("Index Error:", (this_data_length, answer, timestep))
        yield bx, by

### CALLBACK FUNCTIONS ###
def sparse_categorical_perplexity(y_true, y_pred) :
    return K.square(K.sparse_categorical_crossentropy(y_true, y_pred))

class OutputPrediction(Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        output_to_file(model, word_vectors, "output.txt", output_number = 4, max_output_length = OUTPUT_TIME_STEP)
                
def lrate_epoch_decay(epoch) :
    init_lr = LEARNING_RATE
    e = min(LR_DECAY_POW_MAX, (epoch + 1) // LR_DECAY_INTV) # INTV epoches per decay, with max e
    return init_lr * math.pow(LR_DECAY, e)
    
lr_scheduler = LearningRateScheduler(lrate_epoch_decay)
early_stop = EarlyStopping(monitor = "loss", min_delta = EARLYSTOP_MIN_DELTA, patience = EARLYSTOP_PATIENCE)
model_checkpointer = ModelCheckpoint(SAVE_MODEL_NAME)
pred_outputer = OutputPrediction()

### NETWORK MODEL ###
STEPS_PER_EPOCH = int((train_word_count) // BATCH_SIZE * STEP_EPOCH_RATE)

print("\nUSE_SAVED_MODEL:", USE_SAVED_MODEL)
print("max time step:", MAX_TIMESTEP, "\nuse zero offest:", ZERO_OFFSET, "\nrnn units:", RNN_UNIT)
print("\noptimizer:", USE_OPTIMIZER, "\nbatch size:", BATCH_SIZE, "\nstep per epoch:", STEPS_PER_EPOCH, "\nepoches", EPOCHS, "\nlearning_rate:", LEARNING_RATE)
print("validation number:", VALIDATION_NUMBER, "\noutput number:", OUTPUT_NUMBER)

if USE_SAVED_MODEL :
    model = load_model(SAVE_MODEL_NAME)
else :
    sgd = optimizers.SGD(lr = LEARNING_RATE, momentum = 0.5, nesterov = True, decay = 0.0)
    rmsprop = optimizers.RMSprop(lr = LEARNING_RATE, decay = 0.0)
    adam = optimizers.Adam(lr = LEARNING_RATE, decay = 0.0)
    
    if USE_OPTIMIZER == "sgd" :
        optier = sgd
    elif USE_OPTIMIZER == "rmsprop" :
        optier = rmsprop
    elif USE_OPTIMIZER == "adam" :
        optier = adam
    
    ## make model
    input_layer = Input([MAX_TIMESTEP, WV_SIZE])
    rnn_layer = input_layer
    
    for i, v in enumerate(RNN_UNIT) :
        is_return_seq = (i != len(RNN_UNIT) - 1) or USE_ATTENTION or USE_SEQ_RNN_OUTPUT
        if USE_BIDIRECTION :
            if MAX_TIMESTEP :
                rnn_layer = Bidirectional(LSTM(v, return_sequences = is_return_seq))(rnn_layer)
            else :
                rnn_layer = Bidirectional(CuDNNLSTM(v, return_sequences = is_return_seq))(rnn_layer)
        else :
            if MAX_TIMESTEP :
                rnn_layer = LSTM(v, return_sequences = is_return_seq)(rnn_layer)
            else :
                rnn_layer = CuDNNLSTM(v, return_sequences = is_return_seq)(rnn_layer)
    if USE_ATTENTION :
        attention = Dense(1, activation = "softmax")(rnn_layer) # => (?, MAX_TIMESTEP, 1)
        print("attention:", rnn_layer.shape, "to", attention.shape)
        postproc_layer = Multiply()([attention, rnn_layer]) # => (?, MAX_TIMESTEP, last_rnn_units)
        postproc_layer = Lambda(lambda x: K.sum(x, axis = 1))(postproc_layer) # => (?, last_rnn_units)
    elif USE_SEQ_RNN_OUTPUT :
        postproc_layer = Flatten()(rnn_layer)
    else :
        postproc_layer = rnn_layer
    postproc_layer = Dropout(0.1)(postproc_layer)
    guess = Dense(VOCAB_SIZE, activation = "softmax")(postproc_layer)
    model = Model(input_layer, guess)
    model_train = model
    model_train.compile(
        loss = "sparse_categorical_crossentropy", #sparse_categorical_perplexity,
        optimizer = rmsprop,
        metrics = ["sparse_categorical_accuracy"])

gen_train = generate_batch(train_data_list, label_data_list, MAX_TIMESTEP, BATCH_SIZE, ZERO_OFFSET)
gen_test = generate_batch(test_data_list, test_label_data_list, MAX_TIMESTEP, BATCH_SIZE, True)
model_train.summary()
model_train.fit_generator(generator = gen_train,
                    steps_per_epoch = STEPS_PER_EPOCH, 
                    epochs = EPOCHS, 
                    verbose = 1,
                    callbacks = [model_checkpointer, lr_scheduler, early_stop, pred_outputer],
                    validation_data = gen_test, 
                    validation_steps = 1)
model.save(SAVE_MODEL_NAME)
