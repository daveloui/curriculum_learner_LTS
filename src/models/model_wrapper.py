from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from models.conv_net import ConvNet, TwoHeadedConvNet, HeuristicConvNet


class KerasModel ():
    def __init__(self):
        self.mutex = Lock ()
        self.model = None

    def initialize(self, loss_name, search_algorithm, two_headed_model=False, beta=0.1, dropout = 1.0, model_folder='', model_name=''):
        # TODO: the model folder and model name would be where the model is saved

        if (search_algorithm == 'Levin'
                or search_algorithm == 'LevinMult'
                or search_algorithm == 'LevinStar'
                or search_algorithm == 'PUCT'):
            if two_headed_model:
                self.model = TwoHeadedConvNet ((2, 2), 32, 4, loss_name)
            else:
                print("here")
                self.model = ConvNet ((2, 2), 32, 4, loss_name, beta, dropout, model_folder, model_name)
                print("passed")

        if search_algorithm == 'AStar' or search_algorithm == 'GBFS':
            self.model = HeuristicConvNet ((2, 2), 32, 4)

    def predict(self, x):
        with self.mutex:
            return self.model.predict (x)

    def call(self, x):
        return self.model.call(x)

    def train_with_memory(self, memory):
        print ("inside model wrapper train with memory")
        return self.model.train_with_memory (memory)

    def save_weights(self, filepath):
        print ("now saving nn_weights")
        self.model.save_weights (filepath)

    def save_model(self, filepath):
        print ("now saving nn_model")
        self.model.save_weights (filepath)

    def retrieve_output_layer_weights(self):
        print ("now retrieve_output_layer_weights")
        return self.model.retrieve_output_layer_weights()

    def retrieve_output_layer_weights_temp(self):
        print ("now retrieve_output_layer_weights_temp")
        return self.model.retrieve_output_layer_weights_temp()

    def load_weights(self, filepath):
        self.model.load_weights (filepath).expect_partial ()

    def get_gradients_from_batch(self, array_images, array_labels):
        return self.model.get_gradients_from_batch (array_images, array_labels)

    def batch_train_positive_examples(self, training_inputs, training_labels):
        print("calling batch_train_positive_examples")
        return self.model.batch_train_positive_examples(training_inputs, training_labels)

    def clone_model(self):
        return self.model.clone_model()

    def load_weights(self, filepath):
        return self.model.load_weights (filepath).expect_partial ()

    def load_model(self, filepath):
        self.model.load_model(filepath)


class KerasManager (BaseManager):
    pass