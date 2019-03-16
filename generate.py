import os
import random
import h5py
import numpy as np
import train_w2v_model as w2vparas
from keras.models import load_model
from gensim.models import word2vec

OUTPUT_NUMBER = 8
FIXED_PAGE_LENGTH = None # set None to be varied length

model = load_model("rnnstuck_model.h5")
outfile = open("output-generate.txt", "w+", encoding = "utf-8-sig")

if w2vparas.W2V_BY_VOCAB : word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
else : word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
word_vector = word_model.wv
del word_model

WV_SIZE = word_vector.syn0.shape[1]

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

def sample(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, prediction, 1)
    return probas
    
def predict_output_sentence(predict_model, init_sentence = None) :
    output_sentence = [] # zero vector
    if init_sentence :
        output_sentence += init_sentence
    for n in range(200) :
        input_array = make_input_matrix(output_sentence, sentence_length = model.layers[0].input_shape[1])
        y_test = model.predict(input_array)
        y_test = sample(y_test[0], 0.75)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]
        output_sentence.append(next_word)
        if len(output_sentence) > 8 and (next_word == w2vparas.ENDING_MARK_WORD or next_word == w2vparas.ENDING_MARK_CHAR) : break
    output_sentence.append("\n")
    return output_sentence

outfile = open("output-generate.txt", "w+", encoding = "utf-8-sig")
for out_i in range(OUTPUT_NUMBER) :
    init_word = open(w2vparas.CUT_PATH + np.random.choice(w2vparas.PAGENAME_LIST), 'r', encoding = "utf-8-sig").readline().split(" ", 1)[0]
    output_sentence = predict_output_sentence(model, init_sentence = init_word)
    output_string = ""
    for word in output_sentence :
        output_string += word
    outfile.write(output_string)
    outfile.write(">>>>>>>>\n")
    print(out_i)
outfile.close()
