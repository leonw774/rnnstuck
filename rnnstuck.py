import os
import re
import sys
import numpy as np
import random
import h5py
from configure import *
from train_w2v import *
from model import *
from gensim.models import word2vec

# -1 : Use CPU; 0 or 1 : Use GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

### PREPARE TRAINING DATA ###
def make_training_matrices(word_list, word_vectors) :
    input_matrix = np.zeros([1, len(word_list), word_vectors.vectors.shape[1]])
    label_matrix = np.zeros([1, len(word_list), 1], dtype=np.int32)
    
    in_counter = 0
    label_counter = 0

    for word in word_list :
        if word in word_vectors :
            if word != ENDING_MARK and in_counter < len(word_list) :
                input_matrix[0, in_counter] = word_vectors[word]
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

def generate_batch(x, y, max_timestep, batch_size) :
    timestep_size = max_timestep if max_timestep else WORD_LENGTH_MAX
    list_len = len(x)
    bx = np.zeros((batch_size, timestep_size, WV_SIZE))
    by = np.zeros((batch_size, 1), dtype = int)
    while 1 :
        batch_num = np.random.choice(list_len, size=batch_size)
        rands = np.random.random(size = batch_size)
        for i, b in enumerate(batch_num):
            answer = int(rands[i] * x[b].shape[1])
            if answer < max_timestep:
                bx[i, : answer+1] = x[b][:, : answer+1]
                bx[i, answer+1:] = 0
                by[i] = y[b][:, answer]
            else:
                # 'answer' indocate a index of data in the label list (count from 0)
                # a train-label[x] is the one-hot rep of train-data[x+1]
                # so, if answer == 5, the longest train data we can get is [0 : 6], which is 6 in length
                # it means for a timestep, the smallest answer is (timestep - 1)
                # bx from -timestep to end because the last timestep cannot be zero vector
                # x[b]'s last == answer + 1 because it need to include [answer]
                bx[i, :] = x[b][:, answer+1-max_timestep : answer+1]
                by[i] = y[b][:, answer]
        yield (bx, by)

class OutputPrediction(Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        output_to_file(SAVE_MODEL_NAME, "output.txt", output_number = OUTPUT_NUMBER, max_output_timestep = OUTPUT_TIMESTEP)

lr_scheduler = LearningRateScheduler(lrate_epoch_decay)
early_stop = EarlyStopping(monitor = "loss", min_delta = EARLYSTOP_MIN_DELTA, patience = EARLYSTOP_PATIENCE)
model_checkpointer = ModelCheckpoint(SAVE_MODEL_NAME, save_best_only = True)
pred_outputer = OutputPrediction()

if __name__ == "__main__":
    print("\nW2V_BY_VOCAB:", W2V_BY_VOCAB)
    print("WORD_LENGTH_MAX", WORD_LENGTH_MAX)
    print("WORD_LENGTH_MIN", WORD_LENGTH_MIN)
    print("LINE_LENGTH_MAX", LINE_LENGTH_MAX)
    print("LINE_LENGTH_MIN", LINE_LENGTH_MIN)
    if MAX_TIMESTEP :
        if MAX_TIMESTEP > WORD_LENGTH_MIN : print("Warning: WORD_LENGTH_MIN is smaller than MAX_TIMESTEP")
    
    ### PREPARE WORD MODEL ###
    _p, _c = get_train_data(word_min = 2)
    word_model = make_new_w2v(_p, show_result = True)
    word_vectors = word_model.wv
    VOCAB_SIZE = word_vectors.vectors.shape[0]
    del word_model

    page_list, train_word_count = get_train_data(word_min = WORD_LENGTH_MIN, word_max = WORD_LENGTH_MAX, line_min = LINE_LENGTH_MIN, line_max = LINE_LENGTH_MAX)
    random.shuffle(page_list)
    
    print("total word count:", _c)
    print("train word count:", train_word_count)
    
    ### PREPARE TRAINING DATA ###
    train_data_list = []
    label_data_list = []
    for page in page_list :
        t, l = make_training_matrices(page, word_vectors)
        train_data_list.append(t)
        label_data_list.append(l)

    train_test_split = len(page_list) * (VALIDATION_SPLIT-1) // VALIDATION_SPLIT
    test_data_list = train_data_list[train_test_split : ]
    test_label_data_list = label_data_list[train_test_split : ]
    train_data_list = train_data_list[ : train_test_split]
    label_data_list = label_data_list[ : train_test_split]
    
    ### NETWORK MODEL ###
    STEPS_PER_EPOCH = int(train_word_count // BATCH_SIZE * STEP_EPOCH_RATE)
    print("step per epoch: %d\n" % STEPS_PER_EPOCH)

    model = rnnstuck_model(VOCAB_SIZE)

    gen_train = generate_batch(train_data_list, label_data_list, MAX_TIMESTEP, BATCH_SIZE)
    gen_test = generate_batch(test_data_list, test_label_data_list, MAX_TIMESTEP, BATCH_SIZE)
    model.fit_generator(generator = gen_train,
                        steps_per_epoch = STEPS_PER_EPOCH, 
                        epochs = EPOCHS, 
                        verbose = 1,
                        callbacks = [model_checkpointer, lr_scheduler, early_stop, pred_outputer],
                        validation_data = gen_test, 
                        validation_steps = STEPS_PER_EPOCH // (VALIDATION_SPLIT * 10))
    model.save(SAVE_MODEL_NAME)