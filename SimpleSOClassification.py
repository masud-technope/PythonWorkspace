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
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import stem_text

# EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
corpus_file = EXP_HOME + "/small-so/title-answer-code.txt"
corpus_file_label = EXP_HOME + "/small-so/title-answer-code-labels.txt"

fhandler = open(corpus_file, 'r')
flabelHandler = open(corpus_file_label, 'r')

_docs = list()
_labels = list()
for line in fhandler:
    _docs.append(line.strip())
for label in flabelHandler:
    _labels.append(label.strip())

encoder = LabelEncoder()
encoder.fit(_labels)
numLabels = encoder.transform(_labels)
num_of_classes = 3505
# print(numLabels)
one_hot_labels = keras.utils.to_categorical(numLabels, num_classes=num_of_classes)
# print(one_hot_labels)

# quit()

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
ppdocs = list()
for doc in _docs:
    word_list = preprocess_string(doc, CUSTOM_FILTERS)
    ppdocs.append(' '.join(word_list))

vocab_size = 1500
max_length = 20
encoded_docs = [one_hot(d, vocab_size) for d in ppdocs]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs[0])
# X_train = np.array(encoded_docs)
# print(encoded_docs)
# quit()
# now develop the model
input_dim = max_length

model = Sequential()
# model.add(Dense(512, input_shape=(input_dim,)))
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(output_dim=num_of_classes))
model.add(Activation("softmax"))

model.summary()

# one_hot_labels = keras.utils.to_categorical(y_train, num_classes=10)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_docs, one_hot_labels,  batch_size=32, epochs=5, validation_split=0.1, shuffle=True)



