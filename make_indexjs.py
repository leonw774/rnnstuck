import re
import glob
from gensim.models import word2vec

W2V_BY_EACH_WORD = False

index_out = open("wordindex.js", "w+", encoding = 'utf-8-sig')
if W2V_BY_EACH_WORD :
    model = word2vec.Word2Vec.load("myword2vec_by_word.model")
else :
    model = word2vec.Word2Vec.load("myword2vec_by_char.model")
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

seed_out = open("seedindex.js", "w+", encoding = 'utf-8-sig')
cut_posts_paths = glob.glob(r"cut_posts/*.txt")

seed_out.write("var SEED_INDEX = [")
for i, path in enumerate(cut_posts_paths) :
    line = open(path, 'r', encoding = 'utf-8-sig').readline()
    line = (re.sub("= = >", "==>", (re.sub("= = = = = = >", "======>", line))))
    if W2V_BY_EACH_WORD :
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