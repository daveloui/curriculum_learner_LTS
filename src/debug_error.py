import pickle
import numpy as np
import tensorflow as tf

import time
import sys
import os
from os.path import join
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math

from models.memory import Memory, MemoryV2
from models.model_wrapper import KerasManager, KerasModel
from compute_cosines import compute_cosines, compute_rank, retrieve_final_NN_weights #, save_data_to_disk, find_argmax, compute_levin_cost, find_minimum, compute_rank, compute_rank_mins


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed (1)
tf.random.set_seed (1)
p_name = '4x4_979'

def get_image_representation(state_info_dict):
    # obtains all the state data, returns an image object (more on this later)
    assert state_info_dict['puzzle_type'] == 'Witness'
    number_of_colors = state_info_dict['number_of_colors']
    channels = state_info_dict['channels']

    size = state_info_dict['size']
    columns = size[0]
    lines = size[1]

    startPosition = state_info_dict['startPosition']
    column_init = startPosition[0]
    line_init = startPosition[1]

    maxSize = state_info_dict['maxSize']
    max_columns = maxSize[0]
    max_lines = maxSize[1]

    cells = state_info_dict['cells']
    v_seg = state_info_dict['vertical_seg']
    h_seg = state_info_dict['horizontal_seg']

    tips = state_info_dict['tips']
    column_tip = tips[0]
    line_tip = tips[1]

    endPosition = state_info_dict['endPosition']
    column_goal = endPosition[0]
    line_goal = endPosition[1]

    # defining the 3-dimnesional array that will be filled with the puzzle's information
    image = np.zeros ((2 * max_lines, 2 * max_lines, channels))
    # create one channel for each color i
    for i in range (0, number_of_colors):
        for j in range (0, cells.shape[0]):
            for k in range (0, cells.shape[1]):
                if cells[j][k] == i:
                    image[2 * j + 1][2 * k + 1][i] = 1
    channel_number = number_of_colors

    # the number_of_colors-th channel specifies the open spaces in the grid
    for j in range (0, 2 * lines + 1):
        for k in range (0, 2 * columns + 1):
            image[j][k][channel_number] = 1

    # channel for the current path
    channel_number += 1
    for i in range (0, v_seg.shape[0]):
        for j in range (0, v_seg.shape[1]):
            if v_seg[i][j] == 1:
                image[2 * i][2 * j][channel_number] = 1
                image[2 * i + 1][2 * j][channel_number] = 1
                image[2 * i + 2][2 * j][channel_number] = 1

    for i in range (0, h_seg.shape[0]):
        for j in range (0, h_seg.shape[1]):
            if h_seg[i][j] == 1:
                image[2 * i][2 * j][channel_number] = 1
                image[2 * i][2 * j + 1][channel_number] = 1
                image[2 * i][2 * j + 2][channel_number] = 1

    # channel with the tip of the snake
    channel_number += 1
    image[2 * line_tip][2 * column_tip][channel_number] = 1

    # channel for the exit of the puzzle
    channel_number += 1
    image[2 * line_goal][2 * column_goal][channel_number] = 1

    # channel for the entrance of the puzzle
    channel_number += 1
    image[2 * line_init][2 * column_init][channel_number] = 1

    # print("image shape", image.shape)
    return image


def retrieve_batch_data_solved_puzzles(puzzles_list, states, actions, memory_v2):
    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    all_actions = []

    for puzzle_name in puzzles_list:
        print("puzzle_name", puzzle_name)
        images_p = []
        actions_one_hot = tf.one_hot (actions, 4)
        all_actions.append (actions_one_hot)  # a list of tensors
        for state_info_dict in states:
            single_img = get_image_representation (state_info_dict)
            assert isinstance (single_img, np.ndarray)
            images_p.append (single_img)  # a list of np.arrays
        print("finished inner for loop")

        images_p = np.array (images_p) #, dtype=object)  # convert list of np.arrays to an array
        # images_p = np.asarray (images_p).astype ('float32')
        assert isinstance (images_p, np.ndarray)
        print(images_p.shape)
        print("images_p.shape", images_p.shape)
        print("actions_one_hot.numpy ().shape", actions_one_hot.numpy ().shape)

        print("now going to store puzzle images")
        memory_v2.store_puzzle_images_actions (puzzle_name, images_p, actions_one_hot.numpy ())  # TODO: to be used by find min
    return  # TODO: the following used to be uncommented: batch_images, batch_actions


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

flatten_list = lambda l: [item for sublist in l for item in sublist]


file = "solved_puzzles/puzzles_4x4_theta_n-theta_i/BFS_memory_4x4.pkl"
object = open_pickle_file(file)
print("len(obj dict)", len(object[0]))
print("len", len(list(object[0].values())))

for k, v_list in object[0].items():
    if k == p_name:
        v_list_p = v_list
print("len(v_list_p)", len(v_list_p))
states_list = v_list_p[1]
print("number of states", len(states_list))
print("")
actions_list = v_list_p[2]


models_folder = 'trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/'
puzzle_dims = '4x4'
ordering_folder = 'solved_puzzles/puzzles_' + puzzle_dims  # + "_debug_data"

KerasManager.register ('KerasModel', KerasModel)
with KerasManager () as manager:
    nn_model = manager.KerasModel ()
    nn_model.initialize ("CrossEntropyLoss", "Levin", two_headed_model=False)
    nn_weights_file = "trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/pretrained_weights_2.h5"
    nn_model.load_weights (nn_weights_file)
    theta_i = nn_model.retrieve_layer_weights ()

    theta_diff = compute_cosines (nn_model, models_folder, 3)
    print(type(theta_diff))

    memory_v2 = MemoryV2 (ordering_folder, puzzle_dims, "only_add_solutions_to_new_puzzles")
    print("")
    print("calling retrieve_batch_data_solved_puzzles")
    P = [p_name]
    retrieve_batch_data_solved_puzzles (P, states_list, actions_list, memory_v2)

    argmax_p_dot_prods, Rank_dot_prods, argmax_p_cosines, Rank_cosines = compute_rank (P, nn_model, theta_diff,
                                                                                       memory_v2, 4, 1, 1, False)

    print("argmax_p_dot", argmax_p_dot_prods)
    print("Rank_DP", Rank_dot_prods)
    print("argmax_cosines", argmax_p_cosines)
    print("Rank cosines", Rank_cosines)


# theta_diff = compute_cosines (nn_model, self._models_folder, while_loop_iter)




# def store_puzzle_images_and_depths(S, T, memory, checkpoint_folder, subfolder_to_save_ordering, parameters, ncpus=None):
#     # make sure that the data is in the format that we expect it to be:
#     if checkpoint_folder is None: # this is only invoked once, at the beginning
#         # print("checkpoint_folder_is_none")
#         for puzzle, values in S.items ():
#             states_list = values[1]
#             actions_list = values[2]
#             actions_one_hot = tf.one_hot (actions_list, 4)  # num_actions = 4
#             actions_one_hot_array = actions_one_hot.numpy()
#             memory.store_labels (actions_one_hot_array, puzzle)
#
#             num_states = len (states_list)
#             assert num_states == len (actions_list)
#             memory.store_log_depths (puzzle, actions_list)
#
#             for i in range (num_states):
#                 state_dict = states_list[i]  # the first state printed is the state preceding the goal state;
#                 memory.store_image (puzzle, state_dict)  # stores image as np.array
#         print("CHECKPOINT FOLDER IS NONE. WE SAVE ALL IMAGES AND LABELS TO SUBFOLDER")
#
#         memory.save_all_images_depths_labels_to_disk(subfolder_to_save_ordering, parameters)
#         all_keys = S.keys()
#     else:
#         # print("checkpoint folder not none")
#         folder_name = 'all_saved_data/images_depths_labels/' + subfolder_to_save_ordering + '/'
#         filename_imgs = os.path.join (folder_name, 'PuzzleImages_PuzzleFolder' + parameters.puzzle_dims + '_run_' +
#                                       parameters.run_idx + '.pkl')
#         with open (filename_imgs, 'rb') as infile:
#             images_dict = pickle.load (infile)
#         infile.close ()
#
#         filename_imgs_saved_so_far = os.path.join (folder_name, 'ImagesAlreadyStored_PuzzleFolder' +
#                                                    parameters.puzzle_dims + '_run_' + parameters.run_idx + '.pkl')
#         with open (filename_imgs_saved_so_far, 'rb') as infile:
#             images_saved_so_far_dict = pickle.load (infile)
#         infile.close ()
#
#         # open labels file:
#         filename_labels = os.path.join (folder_name, 'labels_PuzzleFolder' + parameters.puzzle_dims + '_run_' +
#                                         parameters.run_idx + '.pkl')
#         with open (filename_labels, 'rb') as infile:
#             labels_dict = pickle.load (infile)
#         infile.close ()
#
#         # open depths file:
#         filename_depths = os.path.join (folder_name, 'depths_PuzzleFolder' + parameters.puzzle_dims + '_run_' +
#                                         parameters.run_idx + '.npy')
#         depths_array = np.load(filename_depths, allow_pickle=True)
#
#         memory.populate_memory (images_dict, images_saved_so_far_dict, labels_dict, depths_array.item())
#         S_keys = list (S.keys ())
#         T_keys = list (T.keys ())
#         all_keys = S_keys + T_keys
#
#     print("Inside store_puzzle_images_and_depths. ncpus =", ncpus)
#     if ncpus is not None:
#         array_images_all, array_actions_all, sum_log_depths_all, p_names_all, dict_idxs = prep_batch_images_labels_log_depths (
#             all_keys, memory, int (parameters.chunksize), ncpus)
#         assert list (dict_idxs.keys ()) == p_names_all  # ensure they are in the same order
#         memory.store_dictionary_of_state_action_idxs (dict_idxs)
#         memory.store_batch_training_images_labels (array_images_all, array_actions_all, sum_log_depths_all)
#     return