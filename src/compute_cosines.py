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


def compute_and_save_cosines_helper_func (theta_diff, grads_P):
    l_theta_diff = []
    l_grads_P = []
    for i, tensor in enumerate(theta_diff):
        assert tensor.shape == grads_P[i].shape
        l_theta_diff.append(tf.keras.backend.flatten(tensor))
        l_grads_P.append(tf.keras.backend.flatten(grads_P[i]))

    theta_diff = tf.concat (l_theta_diff, axis=0)
    grads_P = tf.concat (l_grads_P, axis=0)
    assert theta_diff.shape == grads_P.shape

    dot_prod = tf.tensordot (theta_diff, grads_P, 1).numpy()  # tf.reduce_sum(tf.multiply(a, b))

    theta_diff_l2 = tf.norm (theta_diff, ord=2)
    grads_P_l2 = tf.norm (grads_P, ord=2)
    denom = theta_diff_l2 * grads_P_l2

    cosine = dot_prod / denom.numpy()
    return cosine, dot_prod


def retrieve_final_NN_weights(models_folder):
    # create toy NN:
    print ("models_folder in which we retrieve the weights", models_folder)
    new_model = TempConvNet((2, 2), 32, 4, 'CrossEntropyLoss')
    new_model.load_weights(join (models_folder, "pretrained_weights.h5"))
    theta_n = new_model.retrieve_layer_weights()
    return theta_n


def get_grads_and_CEL_from_batch(array_images, array_labels, theta_model):
    array_images = np.asarray (array_images).astype ('float32')
    array_labels = np.asarray (array_labels).astype ('float32')
    grads = theta_model.get_gradients_from_batch (array_images, array_labels)  # the gradient of the batch (dimensions n x 4)

    # num_images = array_images.shape[0]
    # sum_loss_val = num_images * mean_loss_val
    return grads  # sum_loss_val, mean_loss_val, grads


def compute_cosines(batch_images_P, batch_actions_P, theta_model, models_folder, parameters):
    assert batch_images_P.shape[0] == batch_actions_P.shape[0]

    grads_P = get_grads_and_CEL_from_batch (batch_images_P, batch_actions_P, theta_model)  # _, _, last_grads_P
    theta_i = theta_model.retrieve_layer_weights()  # shape is (128, 4)
    theta_n = retrieve_final_NN_weights(models_folder)

    # assert len (theta_i) == len (theta_n) == len (grads_P)
    theta_diff = [tf.math.subtract(a_i, b_i, name=None) for a_i, b_i in zip(theta_i, theta_n)]
    # assert len(theta_diff) == len(grads_P)

    cosine, dot_prod = compute_and_save_cosines_helper_func (theta_diff, grads_P)
    return cosine, dot_prod


def save_data_to_disk(data, filename):
    np.save(filename, data)


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
