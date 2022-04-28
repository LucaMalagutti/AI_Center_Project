import fasttext
import fasttext.util
ft = fasttext.load_model('word_vectors/cc.en.300.bin')
print("Loaded 300-dim FastText model")
fasttext.util.reduce_model(ft, 200)
print(ft.get_dimension())
ft.save_model('word_vectors/cc.en.200.bin')
print("Saved 200-dim FastText model")

import gensim
import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
glove_vectors.save_word2vec_format('glove-wiki-gigaword-200.bin', binary=True)
print("Saved 200-dim GloVe model")