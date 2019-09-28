import os
import random
import h5py
import numpy as np
from configure import *
from train_w2v import * 
from keras.models import load_model
from gensim.models import word2vec

def make_input_matrix_for_generate(word_list, word_vectors, max_timestep = None) :
    if max_timestep : # only keep last few words if has max_timestep
        word_list = word_list[ -max_timestep : ]
        input_matrix = np.zeros([1, max_timestep, word_vectors.syn0.shape[1]])
    else :
        input_matrix = np.zeros([1, len(word_list), word_vectors.syn0.shape[1]])
    in_counter = 0 if USE_START_MARK else 1
    for word in word_list :
        if word in word_vectors :
            if word != ENDING_MARK :
                input_matrix[0, in_counter] = word_vectors[word] # add one because zero is masked
                in_counter += 1
        else :
            for c in word :
                if c in word_vectors :
                    input_matrix[0, in_counter] = word_vectors[c]
                    in_counter += 1
                if in_counter >= len(word_list) : break
        if in_counter >= len(word_list) : break
    return input_matrix
    
def sample(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    return np.random.multinomial(1, prediction, 1)
    
def predict_output_sentence(predict_model, word_vectors, max_output_length, initial_input_sentence = None) :
    if initial_input_sentence :
        output_sentence = initial_input_sentence
    elif W2V_BY_VOCAB :
        output_sentence = []
    else :
        output_sentence = ""
    for n in range(max_output_length) :
        input_array = make_input_matrix_for_generate(output_sentence, word_vectors, max_timestep = predict_model.layers[0].input_shape[1])
        y_test = predict_model.predict(input_array)
        y_test = sample(y_test[0], 0.8)
        next_word = word_vectors.wv.index2word[np.argmax(y_test[0])]   
        output_sentence.append(next_word)
        if next_word == ENDING_MARK : break
    output_sentence.append("\n")
    return output_sentence

def output_to_file(predict_model, word_vectors, filename, output_number = 1, max_output_length = 100) :
    seed_list, c = get_train_data(word_min = 0, word_max = 2, line_min = 0, line_max = 2)
    outfile = open(filename, "w+", encoding = "utf-8-sig")
    for _ in range(output_number) :
        seed = random.choice(seed_list)[0:2]
        output_sentence = predict_output_sentence(predict_model, word_vectors, max_output_length, seed)
        output_string = ""
        for word in output_sentence :
            output_string += word
        outfile.write(output_string)
        outfile.write(">>>>>>>>\n")
    outfile.close()

if __name__ == "__main__" :
    model = load_model("rnnstuck_model.h5")
    outfile = open("output-generate.txt", "w+", encoding = "utf-8-sig")

    if W2V_BY_VOCAB : word_model = word2vec.Word2Vec.load("myword2vec_by_word.model")
    else : word_model = word2vec.Word2Vec.load("myword2vec_by_char.model")
    word_vectors = word_model.wv
    del word_model
    OUTPUT_NUMBER = 8
    print("MAX_TIMESTEP", model.layers[0].input_shape[1])
    output_to_file(model, word_vectors, "output-generate.txt", 4, 200)

