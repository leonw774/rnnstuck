import os
import random
import h5py
import numpy as np
import process_w2v_model as w2v_paras
from keras.models import load_model
from gensim.models import word2vec

OUTPUT_NUMBER = 8
PAGE_LENGTH = 16 # set None to be varifyed length

ENDING_MARK_WORD = "<e>"
ENDING_MARK_CHAR = '\0'

model = load_model("rnnstuck_model.h5")
outfile = open("output-generate.txt", "w+", encoding = "utf-8-sig")

if w2v_paras.W2V_BY_VOCAB : word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
else : word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
word_vector = word_model.wv
del word_model

WV_SIZE = word_vector.syn0.shape[1]

def make_input_matrix(word_list) :
    input_matrix = np.zeros([1, len(word_list) + 1, WV_SIZE])
    i = 1 # begin at 1 because starting symbol is zero vector
    for word in word_list :
        try :
            input_matrix[0, i] = word_vector[word]
        except KeyError :
            for c in word :
                try :
                    input_matrix[0, i] = word_vector[word]
                except KeyError :
                    continue
        i += 1
    return input_matrix

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
    output_sentence.append(">>>>>>>>\n")
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write("{out_i}\n")
outfile.close()

