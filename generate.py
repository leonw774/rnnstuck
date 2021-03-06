import os
import random
import h5py
import numpy as np
from configure import *
from train_w2v import * 
from tensorflow.keras.models import load_model
from gensim.models import word2vec

def make_input_matrix_for_generate(word_list, word_vectors, max_timestep = None) :
    if max_timestep : # only keep last few words if has max_timestep
        if len(word_list) > max_timestep :
            word_list = word_list[-max_timestep:]
        timestep_size = max_timestep
    else :
        timestep_size = len(word_list)
    input_matrix = np.zeros([1, timestep_size, word_vectors.vectors.shape[1]])
    
    in_counter = 0
    
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
    exp_preds = np.exp(np.log(prediction) / temperature)
    prediction = exp_preds / np.sum(exp_preds)
    return np.random.multinomial(1, prediction, 1)
    
def predict_output_sentence(predict_model, word_vectors, max_output_timestep, seed_sentence = None) :
    if seed_sentence :
        output_sentence = seed_sentence
    elif W2V_BY_VOCAB :
        output_sentence = []
    else :
        output_sentence = ""
    for n in range(max_output_timestep) :
        input_array = make_input_matrix_for_generate(output_sentence, word_vectors, max_timestep = predict_model.layers[0].input_shape[0][1])
        y_test = predict_model.predict(input_array)
        y_test = sample(y_test[0], OUTPUT_SAMPLE_TEMPERATURE)
        next_word = word_vectors.wv.index2word[np.argmax(y_test[0])]
        if W2V_BY_VOCAB :
            output_sentence.append(next_word)
        else :
            output_sentence += next_word
        if next_word == ENDING_MARK: break
    output_sentence += "\n"
    return output_sentence

def output_to_file(predict_model_name, filename, output_number = 1, max_output_timestep = 100):
    word_model = word2vec.Word2Vec.load(W2V_MODEL_NAME)
    word_vectors = word_model.wv
    del word_model
    predict_model = load_model(predict_model_name)
    seed_post_list, c = get_train_data(page_amount = OUTPUT_NUMBER, word_max = 2)
    outfile = open(filename, "w+", encoding = "utf-8-sig")
    for seed_post in seed_post_list:
        output_sentence = predict_output_sentence(predict_model, word_vectors, max_output_timestep, seed_post[:2])
        outfile.write("".join(output_sentence))
        outfile.write(">>>>>>>>\n")
    outfile.close()

if __name__ == "__main__":
    print("MAX_TIMESTEP", model.layers[0].input_shape[1])
    output_to_file("./models/rnnstuck_model.h5", "output-generate.txt", OUTPUT_NUMBER, OUTPUT_TIMESTEP)

