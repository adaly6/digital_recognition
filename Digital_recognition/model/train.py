import tensorflow as tf
from tensorflow.keras.datasets import mnist
from cnn_model import create_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = create_model()

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save('model/saved_model/mnist_cnn.h5')
