import os
import random
import h5py
import numpy as np
from keras.models import load_model
from gensim.models import word2vec


OUTPUT_NUMBER = 10
ENDING_MARK = "<e>"
W2V_BY_EACH_WORD = True
model = load_model("rnnstuck_model.h5")
outfile = open("output-generate.txt", "w+", encoding = "utf-8-sig")

if W2V_BY_EACH_WORD : word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
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
    for n in range(60) :
        y_test = model.predict(make_input_matrix(output_sentence))
        # we only need y_test[0, y_test.shape[1] - 1] because it tells the next missing word
        y_test = sample(y_test[0, y_test.shape[1] - 1], temperature = 0.9)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        if next_word == ENDING_MARK : continue
        output_sentence.append(next_word)
    if out_i % 10 == 0 : print("i:", out_i)
    output_sentence.append("\n")
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write("{out_i}\n")
outfile.close()

