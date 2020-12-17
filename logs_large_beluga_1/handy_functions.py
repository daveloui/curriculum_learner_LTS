import numpy as np
import tensorflow as tf
import os
from os.path import join
import pickle
import math
from tensorflow import keras
from tensorflow.keras import layers
from concurrent.futures.process import ProcessPoolExecutor

from models.model_wrapper import KerasManager, KerasModel
from models.temp_model import TempConvNet #temp_model,

tf.random.set_seed (1)
np.random.seed (1)


def retrieve_final_NN_weights(models_folder, weights_filename="pretrained_weights.h5"):
    # create toy NN:
    print ("models_folder in which we retrieve the weights", models_folder)
    new_model = TempConvNet((2, 2), 32, 4, 'CrossEntropyLoss')
    new_model.load_weights(join (models_folder, weights_filename))
    theta_n = new_model.retrieve_layer_weights()
    return theta_n, new_model


def findMin_helper_function_2 (results):  # results = (puzzle_name, log_frac, states_list, actions_list)
    min_val = float ('inf')
    argmin_p = None
    for result in results:
        new_puzzle_name = result[0]
        frac = result[1]
        if frac < min_val:
            min_val = frac
            argmin_p = new_puzzle_name
    return argmin_p


def findMin_helper_function_1 (data):
    # Note, the trajectories contain the following member variables: states, ..., solution_cost
    theta_model = data[0]
    puzzle_name = data[1]
    memory_model = data[2]
    loss_func = data[3]

    state_images = memory_model.retrieve_puzzle_images(puzzle_name) # list of np.arrays
    log_d = math.log(state_images.shape[0])
    labels = memory_model.retrieve_labels(puzzle_name) # np.array

    # Way #1:
    # action_distribution_log, _ = theta_model.predict (state_images)  #np.array (state_images))
    # elem_wise_mult = np.multiply(labels, action_distribution_log)
    # log_probs = np.sum (elem_wise_mult) # this is the overall log_prob of solving the puzzle given current weights
    # print("log_probs", log_probs)

    # Way #2
    _, _, logits_preds = theta_model.call (state_images)
    CEL = logits_preds.shape[0] * loss_func(labels, logits_preds)
    # print ("CEL loss", CEL)

    log_frac = log_d + CEL  # if uing way #1 --> log_frac = log_d - log_probs
    return puzzle_name, log_frac


def find_minimum(P, theta_model, memory_model, ncpus, chunk_size):
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    with ProcessPoolExecutor (max_workers=ncpus) as executor:
        args = ((theta_model, puzzle_name, memory_model, loss_func) for puzzle_name in P)
        results = executor.map (findMin_helper_function_1, args, chunksize=chunk_size)

    argmin_p = findMin_helper_function_2 (results)
    assert argmin_p is not None
    return argmin_p