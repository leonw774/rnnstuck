import os
import random
import h5py
import numpy as np
import train_w2v as wvparam
from keras.models import load_model
from gensim.models import word2vec

OUTPUT_NUMBER = 8
USE_EMBEDDING = False

model = load_model("rnnstuck_model.h5")
outfile = open("output-generate.txt", "w+", encoding = "utf-8-sig")

if wvparam.W2V_BY_VOCAB : word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
else : word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
word_vector = word_model.wv
del word_model

WV_SIZE = word_vector.syn0.shape[1]
MAX_TIME_STEP = model.layers[0].input_shape[1]
print("MAX_TIME_STEP", MAX_TIME_STEP)

def make_input_matrix(word_list, use_wv = True, sentence_length_limit = None) :
    dim = wvparam.WV_SIZE if use_wv else 1  
    if sentence_length_limit :
        input_matrix = np.zeros([1, sentence_length_limit, dim])
    else :
        input_matrix = np.zeros([1, len(word_list), dim])
    if not use_wv :
        input_matrix = np.squeeze(input_matrix, axis = 2)
    
    if sentence_length_limit :
        word_list = word_list[ -sentence_length_limit : ] # only keep last few words if has sentence_length_limit

    for i, word in enumerate(word_list) :
        try :
            input_matrix[0, i] = word_vector[word] if use_wv else word_vector.vocab[word].index + 1 # add one because zero is masked
        except KeyError :
            for c in word :
                try :
                    input_matrix[0, i] = word_vector[word] if use_wv else word_vector.vocab[word].index + 1
                except KeyError :
                    continue
    return input_matrix
    
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
        input_array = make_input_matrix(output_sentence, use_wv = not USE_EMBEDDING, sentence_length_limit = MAX_TIME_STEP)
        y_test = predict_model.predict(input_array)
        y_test = sample(y_test[0], temperature)
        next_word = word_vector.wv.index2word[np.argmax(y_test[0])]   
        output_sentence.append(next_word)
        if next_word == wvparam.ENDING_MARK : break
    output_sentence.append("\n")
    return output_sentence

def output_to_file(filename, output_number, max_output_length) :
    outfile = open(filename, "w+", encoding = "utf-8-sig")
    if wvparam.W2V_BY_VOCAB :
        init_word = open(wvparam.CUT_PATH + np.random.choice(wvparam.PAGENAME_LIST), 'r', encoding = 'utf-8-sig').readline().split()[0]
    else :
        init_word = open(wvparam.PROC_PATH + np.random.choice(wvparam.PAGENAME_LIST), 'r', encoding = 'utf-8-sig').readline()[0]
    for out_i in range(output_number) :
        output_sentence = predict_output_sentence(model, 0.8, max_output_length, init_word)
        output_string = ""
        for word in output_sentence :
            output_string += word
        outfile.write(output_string)
        outfile.write(">>>>>>>>\n")
    outfile.close()

output_to_file("output-generate.txt", 4, 200)

