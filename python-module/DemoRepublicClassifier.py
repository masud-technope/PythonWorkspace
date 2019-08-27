from gensim.models import Word2Vec
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
tweet_file = EXP_HOME + "/text_classification_data/ExtractedTweets.csv"
corpus_text_file = EXP_HOME + "/text_classification_data/ExtractedTweets.txt"

data_frame = pd.read_csv(tweet_file)

docs = list(data_frame['Tweet'])
CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
ppdocs = list()
for doc in docs:
    word_list = preprocess_string(doc, CUSTOM_FILTERS)
    ppdocs.append(' '.join(word_list))

# out_file = open(corpus_text_file, 'w', encoding="utf-8")
# for pp_doc in ppdocs:
#    out_file.write("%s\n" % pp_doc)

# word_freq = []
# for key, value in d.items():
#    word_freq.append((value, key))

# word_freq.sort(reverse=False)
# print(word_freq)

vocab_size = 12000
max_length = 25
num_of_classes = 433
encoded_docs = [one_hot(d, vocab_size) for d in docs]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs[0])

labels = list(data_frame['Handle'])
encoder = LabelEncoder()
encoder.fit(labels)
numLabels = encoder.transform(labels)
one_hot_labels = keras.utils.to_categorical(numLabels, num_classes=num_of_classes)

input_dim = max_length

model = Sequential()
# model.add(Dense(512, input_shape=(input_dim,)))
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(output_dim=num_of_classes))
model.add(Activation("softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_docs, one_hot_labels, batch_size=32, epochs=5, validation_split=0.1, shuffle=True)


model_file = EXP_HOME + "/text_classification_data/tweet-dem-rep-handle"
model.save(model_file)
print("Trained the model successfully!")

