import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import join
import pickle
import math
import numpy as np
import tensorflow as tf
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
    # flatten_list = lambda l: [item for sublist in l for item in sublist]

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
    outfile = open (filename, 'ab')
    pickle.dump (d, outfile)
    outfile.close ()

    # TODO puzzles_small
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

        # TODO: the following used to be commented
        all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays
        # assert len (all_images_list) == len (all_actions)
        batch_images = np.vstack (all_images_list)
        batch_actions = np.vstack (all_actions)
        assert batch_images.shape[0] == batch_actions.shape[0]

    return batch_images, batch_actions         # TODO: the following used to be commented: batch_images, batch_actions


def map_zero_denom (dot_prod_p_vs_T):
    if dot_prod_p_vs_T < 0.0:
        cosine = float ('-inf')
    elif dot_prod_p_vs_T > 0.0:
        cosine = float ('inf')
    else:
        cosine = float ('nan')
    return cosine


def helper_func(model_name, i):
    theta_i, _ = retrieve_final_NN_weights(models_folder='trained_models_large/BreadthFS_' + model_name, iter=i)
    theta_i_next, _ = retrieve_final_NN_weights (models_folder='trained_models_large/BreadthFS_' + model_name, iter=i+1)
    theta_diff_i = [tf.math.subtract (a_i, b_i, name=None) for a_i, b_i in zip (theta_i, theta_i_next)]

    l_theta_diff_i = []
    for i, tensor in enumerate (theta_diff_i):
        l_theta_diff_i.append (tf.keras.backend.flatten (tensor))

    theta_diff_i = tf.concat (l_theta_diff_i, axis=0)
    theta_diff_i_l2 = tf.norm (theta_diff_i, ord=2).numpy()

    return theta_diff_i_l2


def compute_sum_weights(model_name, ordering_folder):
    pretrained_files = []
    for file in os.listdir('trained_models_large/BreadthFS_' + model_name + '/'):
        if "pretrained" in file:
            pretrained_files += [file]
    pretrained_files.sort()
    num_files = len(pretrained_files)

    summation = 0.0
    for i, filename in enumerate(pretrained_files[:num_files-1]):
        if filename == "pretrained_weights_" + str(i+1) + ".h5":
            theta_diff_i_l2 = helper_func(model_name, i+1)
            summation += theta_diff_i_l2

    np.save(join (ordering_folder, 'Levi_Metric_Denominator.npy'), summation)
    return summation


def compute_and_save_cosines_helper_func (theta_diff, grads_P, label="all_metrics"):
    # grads_P is a list of tensors
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
    new_metric = dot_prod / theta_diff_l2.numpy ()

    grads_P_l2 = tf.norm (grads_P, ord=2)

    # print("theta_diff_l2", theta_diff_l2)
    denom_cosine = theta_diff_l2 * grads_P_l2
    cosine = dot_prod / denom_cosine.numpy()

    if denom_cosine == 0.0:
        print("encountered a zero cosine-denominator!")
        print("cosine =", cosine)

    if theta_diff_l2.numpy () == 0.0:
        print("encountered a zero new_metric-denominator!")
        print("new_metric =", new_metric)

    return cosine, dot_prod, new_metric, grads_P_l2.numpy()


def retrieve_final_NN_weights(models_folder, iter=None): #, weights_filename="pretrained_weights.h5"): # TODO: now we must include the iteration number in the weights
    # Note: if we are in iteration i, that means that the file pretrained_weights_i.h5 contains the weights AFTER training, saved in iteration i
    if iter is None:
        weights_filename = "Final_weights_NoDebug.h5"  # "pretrained_weights_" + str(iter) + ".h5"
    else:
        weights_filename = "pretrained_weights_" + str(iter) + ".h5"
    full_filename = join (models_folder, weights_filename)
    # print("full filename for saved pretrained or Final weights", full_filename)
    # create toy NN:
    new_model = TempConvNet ((2, 2), 32, 4, 'CrossEntropyLoss')
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


def compute_cosines(theta_model, models_folder, iter=None, debugging=False, batch_images_P=None, batch_actions_P=None,
                    Levi_metric_denominator=None):
    # TODO: inputs used to be: batch_images_P, batch_actions_P, theta_model, models_folder, parameters
    theta_i = theta_model.retrieve_layer_weights()  # shape is (128, 4)
    # print("type(theta_i)", type(theta_i), len(theta_i))
    # for w in theta_i:
    #     print(type(w))
    #     print("w.shape", w.shape)
    theta_n, _ = retrieve_final_NN_weights(models_folder, iter) # retrieves either the final layer weights after all of
                                                                # training is done, or the next layer weights theta_{i+1}
    theta_diff = [tf.math.subtract (a_i, b_i, name=None) for a_i, b_i in zip (theta_i, theta_n)]
    if (batch_actions_P is not None) and (batch_images_P is not None):
        grads_P = get_grads_and_CEL_from_batch (batch_images_P, batch_actions_P, theta_model) # TODO: used to be uncommented  # _, _, last_grads_P
        cosine_P, dot_prod_P, new_metric_P, grads_P_l2 = compute_and_save_cosines_helper_func (theta_diff, grads_P)
        assert len (theta_i) == len (theta_n) == len (grads_P)
        assert len (theta_diff) == len (grads_P)
        Levi_metric = grads_P_l2 / Levi_metric_denominator

    if not debugging:
        return cosine_P, dot_prod_P, new_metric_P, theta_diff, Levi_metric
    return theta_diff, theta_i, theta_n


def findArgMax_helper_2 (results):  # results = (puzzle_name, log_frac, states_list, actions_list)
    max_val = float ('-inf')
    argmax_p = None
    for result in results: # each result is a tuple
        new_puzzle_name = result[0]
        dot_prod = result[1]
        if dot_prod > max_val:
            max_val = dot_prod
            argmax_p = new_puzzle_name
    return argmax_p


def findArgMax_helper_1(data):
    # import tensorflow as tf   # <--- Karim said that I might need this when parallelizing with tensorflow_gpu, but even with this, still didn't work.
    nn_model = data[0]
    puzzle_name = data[1]
    memory_model = data[2]
    theta_diff = data[3]
    # label = data[4]

    state_images = memory_model.retrieve_puzzle_images (puzzle_name)
    labels = memory_model.retrieve_labels (puzzle_name)  # np.array
    grads_p_i = get_grads_and_CEL_from_batch (state_images, labels, nn_model)

    cosine, dot_prod, new_metric, grads_P_l2 = compute_and_save_cosines_helper_func (theta_diff, grads_p_i, "cosine_and_dot_prod")
    return puzzle_name, dot_prod, cosine, new_metric


def find_argmax(P_list, nn_model, theta_diff, memory_model, ncpus, chunk_size, n_P, parallelize=True):
    if parallelize:
        chunk_size_heuristic = math.ceil(n_P / (ncpus * 4))
        with ThreadPoolExecutor (max_workers=ncpus) as executor:
            args = ((nn_model, puzzle_name, memory_model, theta_diff) for puzzle_name in P_list)
            results = list(executor.map (findArgMax_helper_1, args, chunksize=chunk_size_heuristic))
    else:
        results = []
        for puzzle_name in P_list:
            arg = (nn_model, puzzle_name, memory_model, theta_diff)
            puzzle_name, dot_prod = findArgMax_helper_1(arg)
            results += [(puzzle_name, dot_prod)]
    argmax_p = findArgMax_helper_2 (results)
    return argmax_p


def compute_rank (P_list, nn_model, theta_diff, memory_model, ncpus, chunk_size, n_P, parallelize=True):
    if parallelize:
        chunk_size_heuristic = math.ceil (n_P / (ncpus * 4))
        with ThreadPoolExecutor (max_workers=ncpus) as executor:
            args = ((nn_model, puzzle_name, memory_model, theta_diff) for puzzle_name in P_list)
            results = list(executor.map (findArgMax_helper_1, args, chunksize=chunk_size_heuristic))
    else:
        results = []
        for puzzle_name in P_list:
            arg = (nn_model, puzzle_name, memory_model, theta_diff)
            puzzle_name, dot_prod, cosine, new_metric = findArgMax_helper_1 (arg)
            results += [(puzzle_name, dot_prod, cosine, new_metric)]

    indices_dot_prods = list (range (len (results)))
    indices_dot_prods.sort (key=lambda x: results[x][1], reverse=True) # gets indices of sorted list results (sorted in descending order acc to dot_prods)
    R_dot_prods = [(None, 0.0)] * len (indices_dot_prods)  #R = []
    for i, x in enumerate (indices_dot_prods):
        R_dot_prods[i] = results[x][:2]  #R.append(results[x][0])
    argmax_p_dot_prods = R_dot_prods[0]


    indices_cosines = list (range (len (results)))
    indices_cosines.sort (key=lambda x: results[x][2], reverse=True) # gets indices of sorted list results (sorted in descending order)
    R_cosines = [(None, 0.0)] * len (indices_cosines)
    for i, x in enumerate (indices_cosines):
        R_cosines[i] = (results[x][0], results[x][2])  #R.append(results[x][0])
        # R_cosines[i][1] = results[x][2]  # R.append(results[x][0])
    argmax_p_cosines = R_cosines[0]


    indices_new_metric = list (range (len (results)))
    indices_new_metric.sort (key=lambda x: results[x][3], reverse=True) # gets indices of sorted list results (sorted in descending order)
    R_new_metric = [(None, 0.0)] * len (indices_new_metric)
    for i, x in enumerate (indices_new_metric):
        R_new_metric[i] = (results[x][0], results[x][3])
    argmax_p_new_metric = R_new_metric[0]

    return argmax_p_dot_prods, R_dot_prods, argmax_p_cosines, R_cosines, argmax_p_new_metric, R_new_metric


def findMin_helper_function_1 (data):
    import tensorflow as tf

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
    # CEL = num_states * aver_CEL
    levin_cost = log_d + (num_states * aver_CEL.numpy())  # if using way #1 --> log_frac = log_d - log_probs
    # aver_levin_cost = (1.0/num_states) * log_d + aver_CEL

    return puzzle_name, levin_cost  #, CEL, log_d


def compute_rank_mins (P_list, nn_model, memory_model, ncpus, chunk_size, n_P, parallelize=True):
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)
    if parallelize:
        chunk_size_heuristic = math.ceil (n_P / (ncpus * 4))
        with ThreadPoolExecutor (max_workers=ncpus) as executor:
            args = ((nn_model, puzzle_name, memory_model, loss_func) for puzzle_name in P_list)
            results = list(executor.map (findMin_helper_function_1, args, chunksize=chunk_size_heuristic))
    else:
        results = []
        for puzzle_name in P_list:
            arg = (nn_model, puzzle_name, memory_model, loss_func)
            puzzle_name, levin_cost = findMin_helper_function_1 (arg)
            results += [(puzzle_name, levin_cost)]

    indices = list (range (len (results)))
    indices.sort (key=lambda x: results[x][1])
    R = [(None, 0.0)] * len (indices)  #R = []  # R = [(None, 0.0)] * len (indices)
    for i, x in enumerate (indices):
        R[i] = results[x]  #R.append(results[x][0])
    argmin_p = R[0]

    return argmin_p, R


def compute_levin_cost(P_batch_states, P_batch_actions, theta_model):
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    _, _, logits_preds = theta_model.call (P_batch_states)
    # print("logits_preds.shape[0] =", logits_preds.shape[0])
    assert logits_preds.shape[0] == P_batch_states.shape[0]

    aver_CEL = loss_func (P_batch_actions, logits_preds).numpy()  # used for training (this is the training loss)
    CEL = logits_preds.shape[0] * aver_CEL  # same as sum_probabilities

    len_trajectory = math.log(P_batch_states.shape[0])

    levin_score = len_trajectory + CEL
    # aver_levin_cost_1 = ((1.0 / logits_preds.shape[0]) * len_trajectory) + aver_CEL  # same as below, down to 10^-9
    aver_levin_cost = (1.0 / logits_preds.shape[0]) * levin_score

    return levin_score, aver_levin_cost, aver_CEL


def save_data_to_disk(data, filename):
    outfile = open (filename, 'wb')
    # outfile = open (filename, 'ab')

    # if os.path.exists (filename):
    #     outfile = open (filename, 'ab')
    # else:
    #     outfile = open (filename, 'wb')
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


def find_minimum(P, theta_model, memory_model, ncpus, chunk_size, n_P, parallelize=True):
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    if parallelize:
        chunk_size_heuristic = math.ceil (n_P / (ncpus * 4))
        with ProcessPoolExecutor (max_workers=ncpus) as executor:
            args = ((theta_model, puzzle_name, memory_model, loss_func) for puzzle_name in P)
            results = executor.map (findMin_helper_function_1, args, chunksize=chunk_size_heuristic)
    else:
        results = []
        for puzzle_name in P_list:
            arg = (theta_model, puzzle_name, memory_model, loss_func)
            puzzle_name, levin_cost = findMin_helper_function_1 (arg)
            results += [(puzzle_name, levin_cost)]

    argmin_p = findMin_helper_function_2 (results)
    assert argmin_p is not None
    return argmin_p







