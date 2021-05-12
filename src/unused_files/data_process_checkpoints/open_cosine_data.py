import sys
import os
import os.path as path
from os import listdir
from os.path import isfile, join
# sys.path.append("..")

import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema
import tensorflow as tf
from tensorflow import keras

from compute_cosines import retrieve_final_NN_weights
from domains.witness import WitnessState
print("passed")

# parent_path = os.path.join (os.pardir, os.getcwd ())
# print ("os.getcwd()", os.getcwd ())
# print ("os.pardir", os.pardir)
# print ("parent_path", parent_path)
# print(parent_path)
# dict_times = np.load("logs_large_beluga_1/dict_times_puzzles_for_cosine_data_4x4-Witness-CrossEntropyLoss.npy",
#                            allow_pickle=True).item()
# print(type(dict_times))
# print(len(dict_times))
# print("")
#
# cos_array = np.load("logs_large_beluga_1/cosine_data_4x4-Witness-CrossEntropyLoss.npy", allow_pickle=True)
# # print("cos_array =", cos_array)
# print(type(cos_array))
# print(cos_array.shape)
# print("")
#
# dot_array = np.load("logs_large_beluga_1/dot_prod_data_4x4-Witness-CrossEntropyLoss.npy", allow_pickle=True)
# # print("dot_array =", dot_array)
# print(type(dot_array))
# print(len(dot_array))
# print("")


# plot data:
def find_intersect(horiz_line_data, list_values):
    assert horiz_line_data.shape == list_values.shape
    list_of_indices_intersection = []
    for i, y in enumerate(list_values):
        if y >= horiz_line_data[i]:
            list_of_indices_intersection += [i]
    return list_of_indices_intersection


def open_pickle_file(filename):
    objects = []
    with (open (filename, "rb")) as openfile:
        print ("filename", filename)
        while True:
            try:
                objects.append (pickle.load (openfile))
            except EOFError:
                break
    openfile.close ()
    # print ("successfully opened pickle file")
    return objects


def plot_and_get_indexes_where_line_above_threshold(list_values, title_name, filename_to_save_fig, x_label="Timesteps",
                                                    y_lim_upper=None, y_lim_lower=None, x_max=None, x_min=-5):
    plt.figure ()
    x = np.arange(0, len(list_values))
    list_values = np.asarray(list_values)

    # plt.scatter (x, list_values, s=8, alpha=1.0)
    # plt.grid (True)
    # plt.title(title_name)
    # plt.xlabel(x_label)
    # if y_lim_upper is not None and y_lim_lower is not None:
    #     plt.ylim(ymin=y_lim_lower, ymax=y_lim_upper)
    # if x_max is not None and x_min is not None:
    #     plt.xlim(xmin=x_min, xmax=x_max)

    indices_of_peaks = None
    dict_indx_peaks_values = {}
    if "dot-prods" in filename:
        c_max_index = argrelextrema (list_values[0:len(list_values)-1], np.greater, order=10)
        indices_of_peaks = c_max_index[0]  # a numpy array
        peak_values = list_values[indices_of_peaks]  # a numpy array
        assert indices_of_peaks.shape == peak_values.shape
        # plt.scatter (indices_of_peaks, peak_values, s=8, color='r', alpha=1.0)

        print("indices_of_peaks", type(indices_of_peaks), len(indices_of_peaks))
        print ("")
        print("peak_values", type(peak_values), len(peak_values))
        print ("")

        for i, idx in enumerate(indices_of_peaks):
            peak_val = peak_values[i]
            dict_indx_peaks_values[idx] = peak_val

    # old plotting code --------------------
    # horiz_line_data = np.array ([0.95 for _ in range (0, len (list_values))])
    # plt.plot (x, horiz_line_data, 'r')
    # idx = find_intersect(horiz_line_data, list_values)
    # # plt.plot (x[idx], list_values[idx], 'go')
    # end of old plotting code ------------

    # plt.show()
    # plt.savefig (filename_to_save_fig)
    # plt.close ()
    return indices_of_peaks, dict_indx_peaks_values


# retrieve files:
def map_to_title(filename):
    if "cosine" in filename:
        return "Cosine(theta_n - theta_i, grad_{theta_i}(P_new))"
    elif "dot_prod" in filename:
        return "(theta_n - theta_i) * grad_{theta_i}(P_new)"
    else:
        print("Cannot find title for file!")


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


def retrieve_batch_data_solved_puzzles(states_list, actions_list):
    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    all_actions = []
    images_p = []
    actions_one_hot = tf.one_hot (actions_list, 4)
    all_actions.append (actions_one_hot)  # a list of tensors
    for s_dict in states_list:
        single_img = get_image_representation (s_dict)
        assert isinstance (single_img, np.ndarray)
        images_p.append (single_img)  # a list of np.arrays
    images_p = np.array (images_p, dtype=object)  # convert list of np.arrays to an array
    images_p = np.asarray (images_p).astype ('float32')
    assert isinstance (images_p, np.ndarray)
    assert images_p.shape[0] == actions_one_hot.numpy().shape[0]

    return images_p, actions_one_hot


def find_solution(sublist, nn_model, loss_func):
    states_list = sublist[1]
    actions_list = sublist[2]

    # convert states list to np.array()
    batch_images_P, batch_actions_P = retrieve_batch_data_solved_puzzles (states_list, actions_list)
    log_d = math.log(batch_images_P.shape[0])

    _, _, logits_preds = nn_model.call (batch_images_P)
    CEL = logits_preds.shape[0] * loss_func (batch_actions_P, logits_preds)
    log_frac = log_d + CEL
    return log_frac.numpy()


def retrieve_min_cost_puzzles(dict_p_dur_peaks):
    model_name = '4x4-Witness-CrossEntropyLoss'
    models_folder = 'trained_models_large/BreadthFS_' + model_name + "/"
    filename = "Final_weights.h5"
    _, temp_model = retrieve_final_NN_weights(models_folder, filename)
    loss_func = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    object = open_pickle_file("solved_puzzles/puzzles_4x4_beluga1/BFS_memory_4x4.pkl")
    d_trajectories = object[0]

    d_solutions = {}
    d_argmins = {}
    for peak_idx, puzzle_sublist in dict_p_dur_peaks.items():
        d_solutions[peak_idx] = []
        if len(puzzle_sublist) == 1:
            d_argmins[peak_idx] = puzzle_sublist[0]
        else:
            for p in puzzle_sublist:
                if p in d_trajectories.keys():
                    cost_p = find_solution(d_trajectories[p], temp_model, loss_func)
                    # print("cost_p", cost_p)
                    d_solutions[peak_idx].append([p, cost_p])
            # print("d_solutions", d_solutions)
            # print("")

            # find argmin_p for each sublist
            min_cost = float ('inf')
            for sublist in d_solutions[peak_idx]:
                p = sublist[0]
                cost_p = sublist[1]
                if cost_p < min_cost:
                    min_cost = cost_p
            d_argmins[peak_idx] = p
    return d_argmins


def retrieve_p_during_peaks(idx_peaks, d):
    # print("inside retrieve_min_cost_puzzles")
    d_values_array = np.array(list(d.values()))
    list_idx_peaks = list(idx_peaks)
    puzzles_coincide_peaks = d_values_array[list_idx_peaks]

    dict_p_dur_peaks = {}
    for i, sublist in enumerate(puzzles_coincide_peaks):
        idx = idx_peaks[i]
        dict_p_dur_peaks[idx] = sublist
    return dict_p_dur_peaks


def generate_puzzle_images(d_argmins):
    witness_puzzle = WitnessState ()

    my_path = "logs_large_beluga_1/plots_puzzles_4x4/"
    for idx, p_name in d_argmins.items():
        subfolder = join (my_path, str(idx))
        if not os.path.exists (subfolder):
            os.makedirs (subfolder)

        witness_puzzle.read_state("problems/witness/puzzles_4x4/" + p_name)
        img_file = join (subfolder, p_name)
        witness_puzzle.save_figure (img_file)

    pass


onlyfiles = [os.path.join('logs_large_beluga_1/', f) for f in os.listdir('logs_large_beluga_1/') if
             os.path.isfile(os.path.join('logs_large_beluga_1/', f))]
print("onlyfiles", onlyfiles)
print("")
print("os.path.join (os.getcwd (), 'logs_large_beluga_1/)", os.path.join (os.getcwd (), 'logs_large_beluga_1/'))

puzzle_dims = "4x4"
path_to_save_figure = "logs_large_beluga_1/plots_puzzles_" + puzzle_dims + "/"
if not os.path.exists (path_to_save_figure):
    os.makedirs (path_to_save_figure, exist_ok=True)

print("path_to_save_figure =", path_to_save_figure)

processed_dot_prods = False
for f in onlyfiles:
    if ("cosine_data_" in f or "dot_prod" in f) and ("dict" not in f):
        # print("loading ", f)
        loaded_array = np.load(f, allow_pickle=True)
        title = map_to_title(f)
        if "cosine_data_" in f:
            filename = os.path.join(path_to_save_figure, "cosines")
            y_lim_upper = 0.15
            y_lim_lower = 0.0
            idx_cos, dict_idxs_peak_values = plot_and_get_indexes_where_line_above_threshold (loaded_array, title, filename,
                                                                       x_label="Timestep when we solve P_new",
                                                                       y_lim_upper=y_lim_upper,
                                                                       y_lim_lower=y_lim_lower)
        else:
            filename = os.path.join (path_to_save_figure, "dot-prods")
            idx_dot_prod, dict_idxs_peak_values = plot_and_get_indexes_where_line_above_threshold(loaded_array, title, filename,
                                                                           x_label="Timestep when we solve P_new")
            processed_dot_prods = True

    if "dict" in f and processed_dot_prods:
        # print("loading ", f)
        d = np.load (f, allow_pickle=True).item()
        dict_p_dur_peaks = retrieve_p_during_peaks(idx_dot_prod, d)
        print("dict_p_dur_peaks", dict_p_dur_peaks)
        d_argmins = retrieve_min_cost_puzzles(dict_p_dur_peaks)
print("")
print("d_argmins =", d_argmins)
print("")
print("dict_idxs_peak_values =", dict_idxs_peak_values)

generate_puzzle_images(d_argmins)


# generate_puzzle_images (puzzles_cos_peaks)



