from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.decomposition import PCA
# from matplotlib import pyplot
import numpy as np
import numpy.core as npc
import keras
from keras_preprocessing.text import one_hot
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Activation
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras import utils
import pandas as pd
from collections import Counter
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text

# EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
corpus_text_file = EXP_HOME + "/stackoverflow/java-question-title/master-java-title.txt"

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
ppdocs = list()
docs = open(corpus_text_file, 'r', encoding="utf-8")

count = 0

for doc in docs:
    count = count + 1
    word_list = preprocess_string(doc, CUSTOM_FILTERS)
    ppdocs.append(' '.join(word_list))

print(count)

model = FastText(ppdocs, size=100, window=5, min_count=5, workers=8)
model_file = EXP_HOME + "/stackoverflow/java-question-title/master-java-title-model"
# model.wv['search']
model.save(model_file)
print("FastText model saved successfully!")
