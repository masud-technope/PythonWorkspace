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
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import SpatialDropout1D
from keras.layers import GlobalMaxPool1D
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
from sklearn import model_selection
from sklearn.preprocessing import MultiLabelBinarizer
from keras.optimizers import SGD
import operator
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import math
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import plot_model


def get_text_version(encoded, item_dict, predicted):
    my_label = list()
    my_dict = dict()
    for index in range(0, len(encoded)):
        my_dict[index] = encoded[index]
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1), reverse=True)
    for key, value in sorted_dict:
        if predicted == True:
            api = item_dict.classes_[key]
            my_label.append(api)
            if len(my_label) == 5:
                break
        else:
            if value > 0.5:
                api = item_dict.classes_[key]
                my_label.append(api)
                if len(my_label) == 5:
                    break
    return my_label


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rrank(x):
    return 1.0 / x


def get_class_weights(all_labels, unique_labels):
    label_dict = dict()
    for label in all_labels:
        if label_dict.__contains__(label):
            label_dict[label] = label_dict[label] + 1
        else:
            label_dict[label] = 1

    my_class_weight = dict()
    for key, value in label_dict.items():
        # score = 1.0 / value
        score = rrank(value)
        my_class_weight[key] = score

    class_weights_now = dict()
    index = 0
    for key in unique_labels:
        value = my_class_weight[key]
        class_weights_now[index] = value
        index = index + 1
    return class_weights_now


def evaluate_error(model):
    count = 0
    predicted_Y = model.predict(X_test, batch_size=128, verbose=2)
    for i in range(len(X_test)):
        predicted = get_text_version(predicted_Y[i], mlb, True)
        actual = get_text_version(Y_test[i], mlb, False)
        for item in predicted:
            if actual.__contains__(item):
                count = count + 1
                break
    print(count / len(X_test))


if __name__ == "__main__":
    EXP_HOME = "C:/My MSc/ThesisWorks/BigData_Code_Search/DeepGenQR/experiment"
    # EXP_HOME = "F:/MyWorks/Thesis Works/Crowdsource_Knowledge_Base/DeepGenQR/experiment"
    corpus_file = EXP_HOME + "/stackoverflow/so-question-api-tagged-50000.csv"
    # corpus_file = EXP_HOME + "/br-corpus/br-corpus-master-all3.csv"

    df = pd.read_csv(corpus_file, encoding='latin-1')

    df = df[df["Title"].notnull()]

    _docs = list(df['Title'])
    _labels = list(df['Tag'])

    all_labels = list()
    all_single_labels = list()
    for item in _labels:
        multil_lables = list()
        for label in item.split(' '):
            multil_lables.append(label)
            all_single_labels.append(label)
        all_labels.append(multil_lables)

    mlb = MultiLabelBinarizer()
    # print(all_single_labels)
    no_of_classes = len(set(all_single_labels))
    print(no_of_classes)
    mlb.fit(all_labels)
    # print(len(mlb.classes_))
    numLabels = mlb.transform(all_labels)
    # print(len(numLabels[0]))

    temp_class_weights = get_class_weights(all_single_labels, mlb.classes_)

    print(temp_class_weights)

    # encoding the documents
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
    ppdocs = list()

    for doc in _docs:
        word_list = preprocess_string(doc, CUSTOM_FILTERS)
        ppdocs.append(' '.join(word_list))
    vocab_size = 5000
    max_length = 15
    num_epochs = 100
    encoded_docs = [one_hot(d, vocab_size) for d in ppdocs]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # create model I
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Activation("relu"))
    model.add(Dropout(rate=0.20))
    model.add(Flatten())
    model.add(Dense(units=len(mlb.classes_)))
    model.add(Activation("sigmoid"))

    model_name = "simple-mlp"

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['top_k_categorical_accuracy'])
    filepath = 'weights/' + model_name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
                                 mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=512)
    history = model.fit(x=padded_docs, y=numLabels, class_weight=temp_class_weights, batch_size=512, epochs=num_epochs,
                        verbose=1, callbacks=[checkpoint, tensor_board], validation_split=0.1, shuffle=True)
