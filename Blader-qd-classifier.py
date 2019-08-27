
import pandas as pd
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import SimpleRNN, TimeDistributed

import imblearn

csv_data=pd.read_csv("data/blader-data.csv")
features=csv_data[0:21]
classes=csv_data['qclass']

X_train=features
Y_train=classes

input_dim=X_train.shape[1]

model = Sequential()
model.add(Dense(64, input_shape=(input_dim,)))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

model.summary()

one_hot_labels = keras.utils.to_categorical(Y_train, num_classes=2)
print(one_hot_labels)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, one_hot_labels,  batch_size=32)

model.summary()



















