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
corpus_file = EXP_HOME + "/br-corpus/br-corpus-master.csv"
df = pd.read_csv(corpus_file, encoding='latin-1')

_docs = list(df['Content'])
_labels = list(df['Keyword'])

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
ppdocs = list()

# unique_tokens = set()
# for _doc in _labels:
#     for token in _doc.split(sep=' '):
#         unique_tokens.add(token)
# print(len(unique_tokens))
# quit()

# preparing Y labels
# encoder = LabelEncoder()
# encoder.fit(_labels)
# numLabels = encoder.transform(_labels)
num_of_classes = 1130

all_labels = list()
for item in _labels:
    multil_lables = list()
    for label in item.split(' '):
        multil_lables.append(label)
    all_labels.append(multil_lables)

mlb = MultiLabelBinarizer()
print(all_labels)
mlb.fit(all_labels)
# print(len(mlb.classes_))
numLabels = mlb.transform(all_labels)
print(len(numLabels[0]))

# print(numLabels)
# one_hot_labels = keras.utils.to_categorical(numLabels, num_classes=num_of_classes)
# print(one_hot_labels)
# quit()

CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
ppdocs = list()

for doc in _docs:
    word_list = preprocess_string(doc, CUSTOM_FILTERS)
    ppdocs.append(' '.join(word_list))

vocab_size = 4000
max_length = 30
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
model.add(Dropout(rate=0.5))
model.add(Flatten())
# model.add(LSTM(128))
model.add(Activation("relu"))
model.add(Dense(output_dim=num_of_classes))
model.add(Activation("sigmoid"))

# model.summary()

# one_hot_labels = keras.utils.to_categorical(y_train, num_classes=10)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# model.fit(padded_docs, numLabels, batch_size=32, epochs=5, validation_split=0.2, shuffle=True)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(padded_docs, numLabels, test_size=0.20)

model.fit(X_train, Y_train, batch_size=64, epochs=5)

predicted_Y = model.predict_classes(X_test, batch_size=64, verbose=1)


# predicted_Y[predicted_Y >= 0.5] = 1
# predicted_Y[predicted_Y < 0.5] = 0


def get_text_version(encoded, item_dict):
    my_label = list()
    for index in range(0, len(encoded)):
        if encoded[index] > 0:
            my_label.append(item_dict.classes_[index])
    return my_label


# predicted class
print("Predicted class:")
# print(mlb.classes_[108])

count = 0
for i in range(len(X_test)):
    print("X=%s, Predicted=%s, Actual=%s" % (
        X_test[i],  mlb.classes_[predicted_Y[i]],  get_text_version(Y_test[i])))
    count = count + 1
    if count == 100:
        break

# saving the model
# model_file = EXP_HOME + "/stackoverflow/deep-model/so-title-api-tagged-50000-v2"
# model.save(model_file)
# print("Saved the NN model")
