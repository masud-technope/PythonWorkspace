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

# define training data
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#              ['this', 'is', 'the', 'second', 'sentence'],
#              ['yet', 'another', 'sentence'],
#              ['one', 'more', 'sentence'],
#              ['and', 'the', 'final', 'sentence']]
# # train model
# model = Word2Vec(sentences, min_count=1)
# # fit a 2D PCA model to the vectors
# X = model[model.wv.vocab]
# print(X)
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
# EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
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

# define class labels
# labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# labels = np.array(_labels)
# print(labels)

encoder = LabelEncoder()
encoder.fit(_labels)
numLabels = encoder.transform(_labels)
one_hot_labels = keras.utils.to_categorical(numLabels, num_classes=603)

# print("My labels")
# print(numLabels)
# print(type(numLabels))

vocab_size = 250
encoded_docs = [one_hot(d, vocab_size) for d in _docs]
print(encoded_docs)

# # pad documents to a max length of 4 words
# max_length = 20
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
#
# # define the model
model = Sequential()
# model.add(Embedding(vocab_size, 512, input_length=max_length))
# model.add(Flatten())

model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
# # compile the model

# # summarize the model
# print(model.summary())

# convert te label to one-hot rep
# num_classes = np.max(numLabels) + 1
# num_classes_encoded = utils.to_categorical(numLabels, num_classes)

# print(max(numLabels))

# currentLabels = npc.array(numLabels).reshape(602,)


# now fit the model
model.add(Dense(603, activation='softmax'))
# model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# # fit the model
model.fit(encoded_docs, one_hot_labels, epochs=50, verbose=0)
# # evaluate the model

model.summary()

loss, accuracy = model.evaluate(encoded_docs, one_hot_labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
