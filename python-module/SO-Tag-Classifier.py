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


def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    return model


def create_new_model(model_input_shape, model_name):
    # create model I
    model1 = Sequential()
    model1.add(Embedding(vocab_size, 100, input_shape=model_input_shape))
    model1.add(Activation("relu"))
    model1.add(Dropout(rate=0.20))
    model1.add(Flatten())
    model1.add(Dense(units=len(mlb.classes_)))
    model1.add(Activation("sigmoid"))
    return model1


def create_new_cnn_model(model_input_shape, model_name):
    model2 = Sequential()
    model2.add(Embedding(vocab_size, 100, input_shape=model_input_shape))
    # using the Convolution network model
    model2.add(SpatialDropout1D(0.2))
    model2.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model2.add(GlobalMaxPool1D())
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(units=len(mlb.classes_)))
    model2.add(Activation("sigmoid"))
    return model2


def train_and_compile(model, model_name):
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['top_k_categorical_accuracy'])
    filepath = 'weights/' + model_name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
                                 mode='auto', period=1)
    tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=512)
    history = model.fit(x=x_train, y=y_train, batch_size=512, epochs=num_epochs, verbose=1,
                        callbacks=[checkpoint, tensor_board], validation_split=0.2)
    return history


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
    corpus_file = EXP_HOME + "/stackoverflow/so-title-api-tagged-50000-ref.csv"
    # corpus_file = EXP_HOME + "/br-corpus/br-corpus-master-all3.csv"

    df = pd.read_csv(corpus_file, encoding='latin-1')

    df = df[df["Title"].notnull()]

    _docs = list(df['Title'])
    _labels = list(df['Tag'])

    # _docs = list(df['Content'])
    # _labels = list(df['Keyword'])

    # print(df['Tag'].value_counts())
    # quit()

    # print(df['Tag'])
    # unique_tokens = set()
    # for _doc in _docs:
    #     for token in _doc.split(' '):
    #         unique_tokens.add(token)
    # # print(unique_tokens)
    # print(len(unique_tokens))
    # quit()

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

    # quit()

    # encoding the class labels
    # encoder = LabelEncoder()
    # encoder.fit(_labels)
    # numLabels = encoder.transform(_labels)
    # num_of_classes = no_of_classes  # len(unique_tokens)
    # one_hot_labels = keras.utils.to_categorical(numLabels, num_classes=num_of_classes)
    # print(len(one_hot_labels[0]))

    temp_class_weights = get_class_weights(all_single_labels, mlb.classes_)

    print(temp_class_weights)

    # encoding the documents
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_punctuation, remove_stopwords, stem_text]
    ppdocs = list()

    for doc in _docs:
        word_list = preprocess_string(doc, CUSTOM_FILTERS)
        ppdocs.append(' '.join(word_list))
    vocab_size = 10000
    max_length = 30  # 15
    encoded_docs = [one_hot(d, vocab_size) for d in ppdocs]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(padded_docs, numLabels, test_size=0.33)

    # print(numLabels)
    # print(one_hot_labels)
    # quit()
    # print(padded_docs[0])
    # X_train = np.array(encoded_docs)
    # print(encoded_docs)
    # quit()
    # now develop the model

    # input_dim = max_length
    #
    # # create model I
    # model1 = Sequential()
    # # model.add(Dense(512, input_shape=(input_dim,)))
    # model1.add(Embedding(vocab_size, 100, input_length=max_length))
    # model1.add(Activation("relu"))
    # model1.add(Dropout(rate=0.25))
    # model1.add(Flatten())
    # model1.add(Dense(units=len(mlb.classes_)))
    # model1.add(Activation("sigmoid"))
    # model1.summary()

    # # create model II
    # model2 = Sequential()
    # model2.add(Embedding(vocab_size, 100, input_length=max_length))
    # # using the Convolution network model
    # model2.add(SpatialDropout1D(0.2))
    # model2.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    # model2.add(GlobalMaxPool1D())
    # model2.add(Dense(256, activation='relu'))
    # model2.add(Dropout(0.2))
    # model2.add(Dense(units=len(mlb.classes_)))
    # model2.add(Activation("sigmoid"))
    #
    # model2.summary()

    # quit()

    # one_hot_labels = keras.utils.to_categorical(y_train, num_classes=10)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # now train/fit the model

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(padded_docs, numLabels, test_size=0.20)
    _ = create_new_model(X_train.shape, "mlp")
    _ = create_new_cnn_model(X_train.shape, "cnn")
    quit()

    # filepath1 = 'weights/' + 'model1' + '.{epoch:02d}-{loss:.2f}.hdf5'
    # checkpoint1 = ModelCheckpoint(filepath1, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
    #                               mode='auto', period=1)
    # tensor_board1 = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    #
    # filepath2 = 'weights/' + 'model2' + '.{epoch:02d}-{loss:.2f}.hdf5'
    # checkpoint2 = ModelCheckpoint(filepath2, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
    #                               mode='auto', period=1)
    # tensor_board2 = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=32)
    #
    # model1.compile(loss='binary_crossentropy', optimizer="adam", metrics=['top_k_categorical_accuracy'])
    # model1.fit(X_train, Y_train, batch_size=128, epochs=1, callbacks=[checkpoint1, tensor_board1],
    #            class_weight=temp_class_weights)
    # evaluate_error(model1)
    #
    # model2.compile(loss='binary_crossentropy', optimizer="adam", metrics=['top_k_categorical_accuracy'])
    # model2.fit(X_train, Y_train, batch_size=128, epochs=1, callbacks=[checkpoint2,tensor_board2], class_weight=temp_class_weights)
    # evaluate_error(model2)
    #
    # quit()

    # print(X_train.shape)
    # quit()

    #model3 = Sequential()
    #model4 = Sequential()

    model3.load_weights('weights/model1.01-0.00.hdf5')
    model4.load_weights('weights/model2.01-0.00.hdf5')

    models = [model3, model4]

    input_shape = X_train.shape

    model_input = Input(shape=input_shape)

    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    print(y)
    ensemble_model = Model(model_input, y, name='ensemble')

    evaluate_error(ensemble_model)

    # quit()

    # score = model.evaluate(X_test, Y_test, batch_size=64)
    # print(score)
    # quit()
    # predicted_Y = model.predict(X_test, batch_size=128, verbose=2)

    # predicted_Y[predicted_Y >= 0.5] = 1
    # predicted_Y[predicted_Y < 0.5] = 0

    # print(predicted_Y[0])

    # predicted class
    # print("Predicted class:")
    # print(mlb.classes_[1104])

    count = 0
    for i in range(len(X_test)):
        predicted = get_text_version(predicted_Y[i], mlb, True)
        actual = get_text_version(Y_test[i], mlb, False)
        for item in predicted:
            if actual.__contains__(item):
                count = count + 1
                break
    print(count / len(X_test))

    # plot_model(model, to_file="keras-so-model.png")

    # for i in range(len(X_test)):
    #     print("X=%s, Predicted=%s, Actual=%s" % (
    #         X_test[i], get_text_version(predicted_Y[i], mlb, True), get_text_version(Y_test[i], mlb, False)))
    #     count = count + 1
    #     if count == 100:
    #         break

    # classification_report(Y_test,predicted_Y)
    # print(confusion_matrix(Y_test, predicted_Y))

    # print(Y_train)
    # for encoded in Y_train:
    #     my_label = list()
    #     for index in range(0, len(encoded)):
    #         if encoded[index] > 0:
    #             my_label.append(mlb.classes_[index])
    #     print(my_label)
    #     break
    # quit()

    # score = model.evaluate(X_test, Y_test, batch_size=64)
    # print(score)
    # Y_pred = model.predict(X_test, batch_size=64, verbose=1)

    # print(Y_pred)
    # print(Y_test)

    # saving the model
    # model_file = EXP_HOME + "/stackoverflow/deep-model/so-title-api-tagged-50000-v2"
    # model.save(model_file)
    # print("Saved the NN model")
