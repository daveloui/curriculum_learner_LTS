import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.loss_functions import CrossEntropyLoss

tf.random.set_seed (1)
np.random.seed (1)

class TempConvNet (tf.keras.Model):

    def __init__(self, kernel_size, filters, number_actions, loss_name, reg_const=0.001):
        tf.keras.backend.set_floatx ('float64')

        super (TempConvNet, self).__init__ (name='Temp')

        self._max_grad_norms = []

        self._reg_const = reg_const
        self._kernel_size = kernel_size
        self._filters = filters
        self._number_actions = number_actions
        self._loss_name = loss_name

        self.conv1 = tf.keras.layers.Conv2D (filters,
                                             kernel_size,
                                             name='conv1',
                                             activation='relu',
                                             dtype='float64')
        # self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1', dtype='float64')
        self.conv2 = tf.keras.layers.Conv2D (filters,
                                             kernel_size,
                                             name='conv2',
                                             activation='relu',
                                             dtype='float64')
        # self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2', dtype='float64')
        self.flatten = tf.keras.layers.Flatten (name='flatten1', dtype='float64')
        self.dense1 = tf.keras.layers.Dense (128,
                                             name='dense1',
                                             activation='relu',
                                             dtype='float64')
        self.dense2 = tf.keras.layers.Dense (number_actions,
                                             name='dense2',
                                             dtype='float64')

        # Build the NN
        inputs = tf.keras.Input (shape=(22, 22, 9,), name="img")
        x = self.conv1 (inputs)
        x = self.conv2 (x)
        x = self.flatten (x)
        x = self.dense1 (x)
        outputs = self.dense2 (x)

        super (TempConvNet, self).__init__ (inputs=inputs, outputs=outputs, name='')
        # print ("len(self.weights) =", len (self.weights))
        # print("len(self.layers)", len(self.layers))
        # print("outputs.shape", outputs.shape)
        # print("")
        # print ("summary for model we use for training", self.summary ())

        self.optimizer = tf.keras.optimizers.Adam (learning_rate=0.0001)  #0.0001

        if loss_name == 'CrossEntropyLoss':
            self._loss_function = CrossEntropyLoss ()

    def predict(self, x):
        log_softmax, x_softmax, _ = self.call (x)
        return log_softmax.numpy (), x_softmax.numpy ()

    def call(self, input_tensor):

        x = self.conv1 (input_tensor)
        #         x = self.pool1(x)
        x = self.conv2 (x)
        #         x = self.pool2(x)
        x = self.flatten (x)
        x = self.dense1 (x)
        logits = self.dense2 (x)
        x_softmax = tf.nn.softmax (logits)
        x_log_softmax = tf.nn.log_softmax (logits)

        return x_log_softmax, x_softmax, logits

    def retrieve_layer_weights(self):
        model_weights = self.weights
        # print ("type(self.weights)", type (self.weights))
        # print ("len (self.weights)", len (self.weights))
        # print ("type(self.trainable_variables)", type (self.trainable_variables))
        # print ("len(self.trainable_variables)", len (self.trainable_variables))
        return model_weights