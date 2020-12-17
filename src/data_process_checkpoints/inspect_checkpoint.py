import tensorflow as tf
import numpy as np
import os
from os.path import join
from tensorflow import keras
from tensorflow.keras import layers
import threading

from models.model_wrapper import KerasManager, KerasModel
from models.temp_model import TempConvNet #temp_model,

tf.random.set_seed (1)
np.random.seed (1)


def retrieve_final_NN_weights(models_folder, weights_filename="pretrained_weights.h5"):
    # create toy NN:
    new_model = TempConvNet((2, 2), 32, 4, 'CrossEntropyLoss')
    full_filename = join (models_folder, weights_filename)
    new_model.load_weights(full_filename)
    theta_n = new_model.retrieve_layer_weights()
    return theta_n, new_model


# tODO: 1. re-solve allpuzzles using epsilon=True, but without computing cosines
# todo con't: this should be easy enough -- just submit levi's code using the h-levin repo
# todo con't: 2. take a look at the final weights

#2.
# models_folder = 'trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/'
onlyfolders = [os.path.join('trained_models_large/', d) for d in os.listdir('trained_models_large/') if
             os.path.isdir(os.path.join('trained_models_large/', d))]
print("onlyfolders =", onlyfolders)
print("")

for folder in onlyfolders:
    print ("folder =", folder)
    theta_n, _ = retrieve_final_NN_weights(folder)
    last_layer_bias_tensor = theta_n[-1]

    print("last_layer_bias_tensor =", last_layer_bias_tensor)
    print("")