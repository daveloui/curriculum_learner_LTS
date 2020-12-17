import numpy as np
import tensorflow as tf
import os
from os.path import join
import pickle
import math
from tensorflow import keras
from tensorflow.keras import layers
import concurrent.futures
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from models.model_wrapper import KerasManager, KerasModel
from models.temp_model import TempConvNet #temp_model,

tf.random.set_seed (1)
np.random.seed (1)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("passed -- tf.config.experimental.set_memory_growth")
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print("did not pass -- tf.config.experimental.set_memory_growth")
    pass


# functions not used:
def get_batch_data_all_puzzles(dict):
    # get array of images for each puzzle
    # get array of labels for each puzzle
    # store
    images = []
    for p_name, state in dict.items():
        image = state.get_image_representation ()
        actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
        state.get_image_representation ()
    pass


def open_pickle_file(filename):
    objects = []
    with (open (filename, "rb")) as openfile:
        while True:
            try:
                objects.append (pickle.load (openfile))
            except EOFError:
                break
    openfile.close ()
    return objects


def check_if_data_saved(puzzle_dims):
    batch_folder = 'solved_puzzles/' + puzzle_dims + "/"
    if not os.path.exists (batch_folder):
        os.makedirs (batch_folder, exist_ok=True)
        return False
    # print (os.path.abspath (batch_folder))
    filename = 'Solved_Puzzles_Batch_Data'
    filename = os.path.join (batch_folder, filename + '.pkl')
    return os.path.isfile (filename)


def store_batch_data_solved_puzzles(puzzles_list, memory_v2, puzzle_dims):
    flatten_list = lambda l: [item for sublist in l for item in sublist]

    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    # print("store_or_retrieve_batch_data_solved_puzzles -- len(puzzles_list)", len(puzzles_list))
    all_images_list = []
    all_actions = []
    d = {}
    for puzzle_name in puzzles_list:
        images_p = []
        trajectory_p = memory_v2.trajectories_dict[puzzle_name]
        actions_one_hot = tf.one_hot (trajectory_p.get_actions (), 4)
        all_actions.append(actions_one_hot)   # a list of tensors
        for s in trajectory_p.get_states ():
            single_img = s.get_image_representation ()
            assert isinstance(single_img, np.ndarray)
            images_p.append( single_img ) # a list of np.arrays
        images_p = np.array(images_p, dtype=object)  # convert list of np.arrays to an array
        assert isinstance (images_p, np.ndarray)

        d[puzzle_name] = [images_p, actions_one_hot.numpy()]

        all_images_list.append(images_p) # list of sublists; each sublists contain np.arrays
    assert len(all_images_list) == len(all_actions)


    batch_images = np.vstack(all_images_list)
    batch_actions = np.vstack(all_actions)
    assert batch_images.shape[0] == batch_actions.shape[0]

    # TODO: create folder and file name where we are going to save the dictionary d
    batch_folder = 'solved_puzzles/' + puzzle_dims + "/"
    if not os.path.exists (batch_folder):
        os.makedirs (batch_folder, exist_ok=True)
    filename = 'Solved_Puzzles_Batch_Data'
    filename = os.path.join (batch_folder, filename + '.pkl')
    print("len(d)", len(d))
    outfile = open (filename, 'ab')
    pickle.dump (d, outfile)
    outfile.close ()

    # TODO debug
    # objects = open_pickle_file(filename)
    # print(type(objects[0]))
    # for k, v in objects[0].items():
    #     print("k", k, " v[0].shape", v[0].shape, " v[1].shape", v[1].shape)


def retrieve_all_batch_images_actions(d, list_solved_puzzles=None, label="S_minus_P"):
    all_images_list = []
    all_actions = []
    all_images_P = []
    all_actions_P = []
    batch_images = None
    batch_actions = None
    batch_images_P = None
    batch_actions_P = None

    if label == "all_of_S":
        for puzzle_name, v_list in d.items():
            images_p = v_list[0]
            actions_p = v_list[1]
            assert isinstance (images_p, np.ndarray)
            assert isinstance (actions_p, np.ndarray)
            all_actions.append (actions_p)  # a list of tensors
            all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays

    elif label == "P_new":
        for puzzle_name in list_solved_puzzles:
            v_list = d[puzzle_name]
            images_p = v_list[0]
            actions_p = v_list[1]
            assert isinstance (images_p, np.ndarray)
            assert isinstance (actions_p, np.ndarray)
            all_actions_P.append (actions_p)  # a list of tensors
            all_images_P.append (images_p)  # list of sublists; each sublists contain np.arrays

    elif label == "S_minus_P":
        for puzzle_name, v_list in d.items ():
            images_p = v_list[0]
            actions_p = v_list[1]
            assert isinstance (images_p, np.ndarray)
            assert isinstance (actions_p, np.ndarray)

            if puzzle_name in list_solved_puzzles:
                all_actions_P.append (actions_p)  # a list of tensors
                all_images_P.append (images_p)
            else:
                all_actions.append (actions_p)  # a list of tensors
                all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays

    assert len (all_images_P) == len (all_actions_P)
    if len(all_images_P) > 0:
        batch_images_P = np.vstack (all_images_P)
        batch_actions_P = np.vstack (all_actions_P)
        assert batch_images_P.shape[0] == batch_actions_P.shape[0]
        assert isinstance (batch_images_P, np.ndarray)
        assert isinstance (batch_actions_P, np.ndarray)

    assert len (all_images_list) == len (all_actions)
    if len (all_images_list) > 0:
        batch_images = np.vstack (all_images_list)
        batch_actions = np.vstack (all_actions)
        assert batch_images.shape[0] == batch_actions.shape[0]
        assert isinstance (batch_images, np.ndarray)
        assert isinstance (batch_actions, np.ndarray)

    return batch_images, batch_actions, batch_images_P, batch_actions_P


# Functions used:
def retrieve_batch_data_solved_puzzles(puzzles_list, memory_v2):
    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    all_images_list = []
    all_actions = []

    # d = {}
    for puzzle_name in puzzles_list:
        images_p = []
        trajectory_p = memory_v2.trajectories_dict[puzzle_name]
        actions_one_hot = tf.one_hot (trajectory_p.get_actions (), 4)
        all_actions.append (actions_one_hot)  # a list of tensors
        for s in trajectory_p.get_states ():
            single_img = s.get_image_representation ()
            assert isinstance (single_img, np.ndarray)
            images_p.append (single_img)  # a list of np.arrays
        images_p = np.array (images_p, dtype=object)  # convert list of np.arrays to an array
        images_p = np.asarray (images_p).astype ('float32')
        assert isinstance (images_p, np.ndarray)

        memory_v2.store_puzzle_images_actions (puzzle_name, images_p, actions_one_hot.numpy ())  # TODO: to be used by find min
        # d[puzzle_name] = [images_p, actions_one_hot.numpy ()]

        # print("images_p.shape", images_p.shape)
        # print("actions_one_hot.shape", actions_one_hot.shape)
        # assert False
        # print("")

        all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays
        # assert len (all_images_list) == len (all_actions)
        batch_images = np.vstack (all_images_list)
        batch_actions = np.vstack (all_actions)
        assert batch_images.shape[0] == batch_actions.shape[0]

    return batch_images, batch_actions


def map_zero_denom (dot_prod_p_vs_T):
    if dot_prod_p_vs_T < 0.0:
        cosine = float ('-inf')
    elif dot_prod_p_vs_T > 0.0:
        cosine = float ('inf')
    else:
        cosine = float ('nan')
    return cosine


def compute_and_save_cosines_helper_func (theta_diff, grads_P, label="cosine_and_dot_prod"):
    # grads_P is a list of tensors
    # print("")
    # print("inside compute_and_save_cosines_helper_func")
    l_theta_diff = []
    l_grads_P = []
    for i, tensor in enumerate(theta_diff):
        assert tensor.shape == grads_P[i].shape
        l_theta_diff.append(tf.keras.backend.flatten(tensor))
        l_grads_P.append(tf.keras.backend.flatten(grads_P[i]))    # where the error happens
    theta_diff = tf.concat (l_theta_diff, axis=0)
    grads_P = tf.concat (l_grads_P, axis=0)
    assert theta_diff.shape[0] == grads_P.shape
    dot_prod = tf.tensordot (theta_diff, grads_P, 1).numpy ()
    if label == "only_dot_prod":
        return dot_prod

    theta_diff_l2 = tf.norm (theta_diff, ord=2)
    grads_P_l2 = tf.norm (grads_P, ord=2)
    denom = theta_diff_l2 * grads_P_l2

    cosine = dot_prod / denom.numpy()
    print("got cosine and dot_prod")
    return cosine, dot_prod


def retrieve_final_NN_weights(models_folder, weights_filename="pretrained_weights.h5"):
    # create toy NN:
    new_model = TempConvNet((2, 2), 32, 4, 'CrossEntropyLoss')
    full_filename = join (models_folder, weights_filename)
    new_model.load_weights(full_filename)
    theta_n = new_model.retrieve_layer_weights()
    return theta_n, new_model


def get_grads_and_CEL_from_batch(array_images, array_labels, theta_model):
    array_images = np.asarray (array_images).astype ('float32')
    array_labels = np.asarray (array_labels).astype ('float32')
    grads = theta_model.get_gradients_from_batch (array_images, array_labels)  # the gradient of the batch (dimensions n x 4)
    # num_images = array_images.shape[0]
    # sum_loss_val = num_images * mean_loss_val
    return grads  # sum_loss_val, mean_loss_val, grads


def compute_cosines(batch_images_P, batch_actions_P, theta_model, models_folder, parameters):
    # print("inside compute cosines")
    assert batch_images_P.shape[0] == batch_actions_P.shape[0]

    grads_P = get_grads_and_CEL_from_batch (batch_images_P, batch_actions_P, theta_model)  # _, _, last_grads_P
    theta_i = theta_model.retrieve_layer_weights()  # shape is (128, 4)
    theta_n, _ = retrieve_final_NN_weights(models_folder)

    # assert len (theta_i) == len (theta_n) == len (grads_P)
    theta_diff = [tf.math.subtract(a_i, b_i, name=None) for a_i, b_i in zip(theta_i, theta_n)]
    # assert len(theta_diff) == len(grads_P)

    cosine, dot_prod = compute_and_save_cosines_helper_func (theta_diff, grads_P)
    return cosine, dot_prod, theta_diff


def findArgMax_helper_2 (results):  # results = (puzzle_name, log_frac, states_list, actions_list)
    print("inside findArgMax_helper_2")
    print("type(results) =", type(results))
    print("results =", results)
    max_val = float ('-inf')
    argmax_p = None
    for result in results: # each result is a tuple
        print("result", result)
        new_puzzle_name = result[0]
        dot_prod = result[1]
        if dot_prod > max_val:
            max_val = dot_prod
            argmax_p = new_puzzle_name
    return argmax_p


def findArgMax_helper_1(data):
    # print("inside findArgMax_helper_1")
    # compute grads_c(p_i)
    nn_model = data[0]
    puzzle_name = data[1]
    memory_model = data[2]
    theta_diff = data[3]

    state_images = memory_model.retrieve_puzzle_images (puzzle_name)
    labels = memory_model.retrieve_labels (puzzle_name)  # np.array
    grads_p_i = get_grads_and_CEL_from_batch (state_images, labels, nn_model)


    dot_prod = compute_and_save_cosines_helper_func (theta_diff, grads_p_i, "only_dot_prod")
    return puzzle_name, dot_prod


def find_argmax(P_list, nn_model, theta_diff, memory_model, ncpus, chunk_size, n_P):
    print("")
    print("inside find_argmax -- ", P_list)
    s1 = time.time()

    chunk_size_heuristic = math.ceil(n_P / (ncpus * 4))
    with ThreadPoolExecutor (max_workers=ncpus) as executor:
        args = ((nn_model, puzzle_name, memory_model, theta_diff) for puzzle_name in P_list)
        results = list(executor.map (findArgMax_helper_1, args, chunksize=chunk_size_heuristic))
    print("successfully executed parallelization to -- findArgMax_helper_1")
    e1 = time.time()
    time_el1 = e1 - s1
    # print("time_el1 =", time_el1)

    # s2 = time.time()
    # with ThreadPoolExecutor (max_workers=ncpus) as executor:
    #     args = ((nn_model, puzzle_name, memory_model, theta_diff) for puzzle_name in P_list)
    #     result_futures = list (map (lambda x: executor.submit (findArgMax_helper_1, x), args))
    #     results = [f.result () for f in concurrent.futures.as_completed (result_futures)]  # this is a list
    # e2 = time.time()
    # time_el2 = e2 - s2
    # print("time_el2 =", time_el2)

    argmax_p = findArgMax_helper_2 (results)
    return argmax_p


def compute_rank (P_list, nn_model, theta_diff, memory_model, ncpus, chunk_size, n_P):
    # print("inside compute rank")
    chunk_size_heuristic = math.ceil (n_P / (ncpus * 4))
    with ThreadPoolExecutor (max_workers=ncpus) as executor:
        args = ((nn_model, puzzle_name, memory_model, theta_diff) for puzzle_name in P_list)
        results = list(executor.map (findArgMax_helper_1, args, chunksize=chunk_size_heuristic))

    indices = list (range (len (results)))
    indices.sort (key=lambda x: results[x][1], reverse=True)
    # print("sorted_indices", indices)
    R = [(None, 0.0)] * len (indices)  #R = []  # R = [(None, 0.0)] * len (indices)
    for i, x in enumerate (indices):
        # print("results[x] =", results[x])
        R[i] = results[x][0]  #R.append(results[x][0])
    argmax_p = R[0]
    return argmax_p, R


def findMin_helper_function_1 (data):
    theta_model = data[0]
    puzzle_name = data[1]
    memory_model = data[2]
    loss_func = data[3]

    state_images = memory_model.retrieve_puzzle_images(puzzle_name) # list of np.arrays
    labels = memory_model.retrieve_labels(puzzle_name) # np.array
    num_states = state_images.shape[0]
    log_d = math.log (num_states)

    # Way #1:
    # action_distribution_log, _ = theta_model.predict (state_images)  #np.array (state_images))
    # elem_wise_mult = np.multiply(labels, action_distribution_log)
    # log_probs = np.sum (elem_wise_mult) # this is the overall log_prob of solving the puzzle given current weights
    # levin_score = log_d - log_probs
    # average_levin_score = (1.0/num_states) * levin_score

    # Way #2
    _, _, logits_preds = theta_model.call (state_images)
    assert logits_preds.shape[0] == num_states

    aver_CEL = loss_func(labels, logits_preds)   # loss_func(labels, logits_preds) was previously defined as the average CEL!
    CEL = num_states * aver_CEL
    levin_cost = log_d + (num_states * aver_CEL.numpy())  # if using way #1 --> log_frac = log_d - log_probs
    # aver_levin_cost = (1.0/num_states) * log_d + aver_CEL

    return puzzle_name, levin_cost  #, CEL, log_d


def compute_rank_mins (P_list, nn_model, memory_model, ncpus, chunk_size, n_P):
    # print ("inside compute rank_mins")
    chunk_size_heuristic = math.ceil (n_P / (ncpus * 4))
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)
    with ThreadPoolExecutor (max_workers=ncpus) as executor:
        args = ((nn_model, puzzle_name, memory_model, loss_func) for puzzle_name in P_list)
        results = list(executor.map (findMin_helper_function_1, args, chunksize=chunk_size_heuristic))

    indices = list (range (len (results)))
    indices.sort (key=lambda x: results[x][1])
    R = [(None, 0.0)] * len (indices)  #R = []  # R = [(None, 0.0)] * len (indices)
    for i, x in enumerate (indices):
        R[i] = results[x][0]  #R.append(results[x][0])
    argmin_p = R[0]
    return argmin_p, R


def compute_levin_cost(P_batch_states, P_batch_actions, theta_model):
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)
    _, _, logits_preds = theta_model.call (P_batch_states)
    # print("logits_preds.shape[0] =", logits_preds.shape[0])
    assert logits_preds.shape[0] == P_batch_states.shape[0]

    aver_CEL = loss_func (P_batch_actions, logits_preds)  # used for training (this is the training loss)
    CEL = logits_preds.shape[0] * aver_CEL  # same as sum_probabilities

    len_trajectory = math.log(P_batch_states.shape[0])

    levin_score = len_trajectory + CEL
    # aver_levin_cost_1 = ((1.0 / logits_preds.shape[0]) * len_trajectory) + aver_CEL  # same as below, down to 10^-9
    aver_levin_cost = (1.0 / logits_preds.shape[0]) * levin_score

    return levin_score, aver_levin_cost, aver_CEL


def save_data_to_disk(data, filename):
    # np.save(filename, data)
    if not os.path.exists (filename):
        outfile = open (filename, 'wb')
    else:
        outfile = open (filename, 'ab')
    pickle.dump (data, outfile)
    outfile.close ()

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


def find_minimum(P, theta_model, memory_model, ncpus, chunk_size, n_P):
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    chunk_size_heuristic = math.ceil (n_P / (ncpus * 4))
    with ProcessPoolExecutor (max_workers=ncpus) as executor:
        args = ((theta_model, puzzle_name, memory_model, loss_func) for puzzle_name in P)
        results = executor.map (findMin_helper_function_1, args, chunksize=chunk_size_heuristic)

    argmin_p = findMin_helper_function_2 (results)
    assert argmin_p is not None
    return argmin_p







