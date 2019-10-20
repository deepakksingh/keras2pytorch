
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPooling2D, Dropout,\
                                    Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from datetime import datetime

batch_size = 128
epochs = 1


class mnist_cnn(Model):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3), activation="relu")
        self.conv2 = Conv2D(filters=64, kernel_size=(3,3), activation="relu")
        self.max_pool = MaxPooling2D(pool_size=(2,2))
        self.dropout1 = Dropout(0.25)
        self.flatten = Flatten()
        self.fc1 = Dense(units=128, activation="relu")
        self.dropout2 = Dropout(0.5)
        self.fc2 = Dense(units=10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train/255, axis=-1)
    x_test = np.expand_dims(x_test/255, axis=-1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = mnist_cnn()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"],
                  learning_rate=1e-3
                 )

    logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir,
                                       histogram_freq=1,
                                       write_graph=False,
                                       write_grads=True,
                                       write_images=True,
                                       update_freq='batch'
                                      )
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback]
             )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])


if __name__=="__main__":
    main()
