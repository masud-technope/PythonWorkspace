import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras import utils

X_train = np.array([[1, 2], [6, 5], [8, 2]])
y_train = np.array([2, 3, 7])


input_dim = X_train.shape[1]

model = Sequential()
model.add(Dense(64, input_shape=(input_dim,)))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.summary()

one_hot_labels = keras.utils.to_categorical(y_train, num_classes=10)

#print(X_train)
print(one_hot_labels)


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, one_hot_labels,  batch_size=32)
