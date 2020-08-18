import re
import glob
from gensim.models import word2vec

W2V_BY_VOCAB = True

if W2V_BY_VOCAB :
    model = word2vec.Word2Vec.load("./models/myword2vec_by_word.model")
else :
    model = word2vec.Word2Vec.load("./models/myword2vec_by_char.model")
    
### VOCAB INDEX ###
index_out = open("words.js", "w+", encoding = 'utf-8-sig')
index_out.write("var WORD_INDEX = [")
for i in range(model.wv.syn0.shape[0]) :
    word = model.wv.index2word[i];
    if word == '\\' or word == '\"' or word == '\/' :
        word = '\\' + word
    if word == '\n' :
        word = '\\n'
    if i == 0 :
        index_out.write("\"" + word + "\"")
    else :
        index_out.write(", \"" + word + "\"")
index_out.write("];")
print("words done")

### VECTOR INDEX ###
index_out = open("vectors.js", "w+", encoding = 'utf-8-sig')
index_out.write("var VECTOR_DICT = {")
for i in range(model.wv.syn0.shape[0]) :
    word = model.wv.index2word[i];
    printed_word = ""
    if word == '\\' or word == '\"' or word == '\/':
        printed_word = '\\' + word
    if word == '\n':
        printed_word = '\\n'
    if i == 0:
        index_out.write("\"" + printed_word + "\":[" + ",".join([str(x) for x in model.wv[word]]) + "]")
    else :
        index_out.write(", \"" + printed_word + "\":[" + ",".join([str(x) for x in model.wv[word]]) + "]")
index_out.write("};")
print("vectors done")

### SEED ###
seed_out = open("seeds.js", "w+", encoding = 'utf-8-sig')
cut_posts_paths = glob.glob(r"cut_posts/*.txt")

seed_out.write("var SEED_INDEX = [")
for i, path in enumerate(cut_posts_paths) :
    line = open(path, 'r', encoding = 'utf-8-sig').readline()
    line = (re.sub("= = >", "==>", (re.sub("= = = = = = >", "======>", line))))
    if W2V_BY_VOCAB :
        line = line.split()
    if len(line) > 0 :
        word = line[0]
        if word == '\\' or word == '\"' or word == '\/' :
            word = '\\' + word
        if word == '\n' :
            word = '\\n'
        if i == 0 :
            seed_out.write("\"" + word + "\"")
        else :
            seed_out.write(", " + "\"" + word + "\"")
seed_out.write("];")
print("seeds done")
