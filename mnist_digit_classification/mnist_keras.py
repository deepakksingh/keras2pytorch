
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPooling2D, Dropout,\
                                    Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

batch_size = 128
epochs = 1


model = Sequential()
model.add(Input((28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"],
              learning_rate=1e-3
             )


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train/255, axis=-1)
x_test = np.expand_dims(x_test/255, axis=-1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
         )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
