from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
import numpy.core as npc
import keras
from keras_preprocessing.text import one_hot
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from keras import utils
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text
import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn.preprocessing import MultiLabelBinarizer
from keras.optimizers import SGD

# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
java_csv_file = EXP_HOME + "/stackoverflow/tag-test/java.csv"
python_csv_file = EXP_HOME + "/stackoverflow/tag-test/python.csv"
csharp_csv_file = EXP_HOME + "/stackoverflow/tag-test/c#.csv"

java_df = pd.read_csv(java_csv_file)
python_df = pd.read_csv(python_csv_file)
cs_df = pd.read_csv(csharp_csv_file)

java_df['label'] = 'java'
python_df['label'] = 'python'
cs_df['label'] = 'c#'

frames = [java_df, python_df, cs_df]

master_df = pd.concat(frames)
# print(master_df)
# master_df['Content'] = master_df['Title']+' '+master_df['Body']

# print(master_df['Content'])
# quit()

_docs = master_df['Title']
_labels = master_df['label']

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
ppdocs = list()

# preparing the corpus
count = 0
for doc in _docs:
    word_list = preprocess_string(doc, CUSTOM_FILTERS)
    ppdocs.append(' '.join(word_list))
vocab_size = 12000
max_length = 20
encoded_docs = [one_hot(d, vocab_size) for d in ppdocs]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
input_dim = max_length

# preparing Y labels
encoder = LabelEncoder()
encoder.fit(_labels)
numLabels = encoder.transform(_labels)
num_of_classes = 3
one_hot_labels = keras.utils.to_categorical(numLabels, num_classes=num_of_classes)

print("Creating the model ")
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(output_dim=num_of_classes))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_docs, one_hot_labels, batch_size=64, epochs=5, validation_split=0.2, shuffle=True)








