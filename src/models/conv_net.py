import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from models.loss_functions import LevinLoss, CrossEntropyLoss, ImprovedLevinLoss, RegLevinLoss

tf.random.set_seed (1)
np.random.seed (1)


class InvalidLossFunction (Exception):
    pass


class TwoHeadedConvNet (tf.keras.Model):

    def __init__(self, kernel_size, filters, number_actions, loss_name, reg_const=0.001):
        tf.keras.backend.set_floatx ('float64')

        super (TwoHeadedConvNet, self).__init__ (name='')

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
        self.pool1 = tf.keras.layers.MaxPooling2D (pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1',
                                                   dtype='float64')
        self.conv2 = tf.keras.layers.Conv2D (filters,
                                             kernel_size,
                                             name='conv2',
                                             activation='relu',
                                             dtype='float64')
        self.pool2 = tf.keras.layers.MaxPooling2D (pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2',
                                                   dtype='float64')
        self.flatten = tf.keras.layers.Flatten (name='flatten1', dtype='float64')

        # Probability distribution
        self.dense11 = tf.keras.layers.Dense (128,
                                              name='dense11',
                                              activation='relu',
                                              dtype='float64')
        self.dense12 = tf.keras.layers.Dense (number_actions,
                                              name='dense12',
                                              dtype='float64')

        # Heuristic value
        self.dense21 = tf.keras.layers.Dense (128,
                                              name='dense21',
                                              activation='relu',
                                              dtype='float64')
        self.dense22 = tf.keras.layers.Dense (1,
                                              name='dense22',
                                              dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam (learning_rate=0.0001)

        self._loss_name = loss_name

        if loss_name == 'LevinLoss':
            self._loss_function = LevinMSELoss ()
        elif loss_name == 'CrossEntropyLoss':
            self._loss_function = CrossEntropyMSELoss ()
        elif loss_name == 'ImprovedLevinLoss':
            self._loss_function = ImprovedLevinMSELoss ()
        elif loss_name == 'RegLevinLoss':
            self._loss_function = RegLevinMSELoss ()
        else:
            raise InvalidLossFunction

    def predict(self, x):
        log_softmax, x_softmax, _, pred_h = self.call (x)
        return log_softmax.numpy (), x_softmax.numpy (), pred_h.numpy ()

    def call(self, input_tensor):

        x = self.conv1 (input_tensor)
        #         x = self.pool1(x)
        x = self.conv2 (x)
        #         x = self.pool2(x)
        x_flatten = self.flatten (x)
        x1 = self.dense11 (x_flatten)
        logits_pi = self.dense12 (x1)
        x_log_softmax = tf.nn.log_softmax (logits_pi)
        x_softmax = tf.nn.softmax (logits_pi)

        x2 = self.dense21 (x_flatten)
        logits_h = self.dense22 (x2)
        return x_log_softmax, x_softmax, logits_pi, logits_h

    def train_with_memory(self, memory):
        losses = []
        memory.shuffle_trajectories ()
        for trajectory in memory.next_trajectory ():
            with tf.GradientTape () as tape:
                loss = self._loss_function.compute_loss (trajectory, self)

            grads = tape.gradient (loss, self.trainable_weights)
            self.optimizer.apply_gradients (zip (grads, self.trainable_weights))
            losses.append (loss)

        return np.mean (losses)

    def get_number_actions(self):
        return self._number_actions


class ConvNet (tf.keras.Model):

    def __init__(self, kernel_size, filters, number_actions, loss_name, reg_const=0.001, beta=0.1, dropout=1.0, model_folder='', model_name=''):
        tf.keras.backend.set_floatx ('float64')

        super (ConvNet, self).__init__ (name='')

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
        super (ConvNet, self).__init__ (inputs=inputs, outputs=outputs, name='')
        # print ("len(self.weights) =", len (self.weights))
        # print ("summary for model we use for training", self.summary ())

        # self.optimizer = tf.keras.optimizers.Adam (learning_rate=0.0001)  #0.0001
        self.optimizer = tf.keras.optimizers.SGD (learning_rate=0.1)

        if loss_name == 'LevinLoss':
            self._loss_function = LevinLoss ()
        elif loss_name == 'ImprovedLevinLoss':
            self._loss_function = ImprovedLevinLoss ()
        elif loss_name == 'CrossEntropyLoss':
            self._loss_function = CrossEntropyLoss ()
        elif loss_name == 'RegLevinLoss':
            self._loss_function = RegLevinLoss ()
        else:
            raise InvalidLossFunction

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
        # print("x_log_softmax", x_log_softmax)
        # print("x_softmax", x_softmax)
        # print("logits", logits)

        # print("")
        # print("x_softmax", x_softmax)
        # summat = tf.math.reduce_sum(x_softmax)
        # print("num_states =", x_softmax.shape[0], "sum of x_softmax =", summat)
        # assert False
        return x_log_softmax, x_softmax, logits

    def _cross_entropy_loss(self, states, y):
        images = [s.get_image_representation () for s in states]
        _, _, logits = self (np.array (images))
        return self.cross_entropy_loss (y, logits)

    def get_gradients_from_batch(self, batch_training_data, batch_actions):
        # print("inside get_gradients_from_batch")
        with tf.GradientTape () as tape:
            tape.watch (self.trainable_variables)
            x = self.conv1 (batch_training_data)
            x = self.conv2 (x)
            x = self.flatten (x)
            x = self.dense1 (x)
            preds = self.dense2 (x)
            # labels = self.get_label (batch_actions)
            loss = self._loss_function.cross_entropy_loss (batch_actions, preds)
        grads = tape.gradient (loss, self.trainable_variables)
        # print(type(grads), len(grads))
        # loss_val = loss.numpy ()
        return grads

    def train_with_memory(self, memory):
        # print ("inside convnet train_with_memory")
        losses = []
        memory.shuffle_trajectories ()
        for trajectory in memory.next_trajectory ():
            with tf.GradientTape () as tape:
                loss = self._loss_function.compute_loss (trajectory, self)
            grads = tape.gradient (loss, self.trainable_weights)

            self.optimizer.apply_gradients (zip (grads, self.trainable_weights))
            losses.append (loss)
        return np.mean (losses)

    def train(self, states, y):
        with tf.GradientTape () as tape:
            loss = self._cross_entropy_loss (states, y)
        grads = tape.gradient (loss, self.trainable_weights)
        self.optimizer.apply_gradients (zip (grads, self.trainable_weights))

        return loss

    def batch_train_positive_examples(self, batch_training_data, batch_actions): # batch data is a list of traject_data_list(s)
        with tf.GradientTape () as tape:
            tape.watch (self.trainable_variables)
            x = self.conv1 (batch_training_data)
            x = self.conv2 (x)
            x = self.flatten (x)
            x = self.dense1 (x)
            preds = self.dense2 (x)
            # print("here, preds.shape =", preds.shape) # these are logits
            loss = self._loss_function.compute_loss_w_batch (batch_actions, preds)
        grads = tape.gradient (loss, self.trainable_variables)
        self.optimizer.apply_gradients (zip (grads, self.trainable_variables))
        loss_val = loss.numpy ()
        # print("loss_val", loss_val)
        return loss_val

    def train_w_batch(self, batch_training_data, batch_actions, grads_train):
        self.optimizer.apply_gradients (zip (grads_train, self.trainable_variables))
        return

    def retrieve_layer_weights(self):
        model_weights = self.weights
        return model_weights

    def retrieve_output_layer_weights_temp(self):
        print("retrieve_output_layer_weights_temp")
        # output_weights = self.dense2.weights[0].numpy()
        output_weights = self.dense2.weights
        return output_weights

    def clone_model(self):
        new_model = keras.models.clone_model (self)
        print("now we have a new_model")
        print(new_model)


    def get_number_actions(self):
        return self._number_actions

