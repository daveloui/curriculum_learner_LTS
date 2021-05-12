# TODO:
#  For witness puzzles:
#   1. get the top k=5 puzzles using Rank data
#   2. plot each subset of top 5 puzzles for each iteration


import sys
import os
import os.path as path
from os import listdir
from os.path import isfile, join

import argparse
import pickle
import copy
import random
import math
from typing import Dict, Any
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import pandas as pd
import plotly.express as px

from Witness_Puzzle_Image import WitnessPuzzle


def open_pickle_file(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    openfile.close()
    return objects


flatten_list = lambda l: [item for sublist in l for item in sublist]


def plot_data(list_values, idxs_new_puzzles, title_name, filename_to_save_fig, x_label="Number of New Puzzles Solved",
              y_lim_upper=None, y_lim_lower=None, x_max=None, x_min=None, flag=''):
    #TODO: 1. find puzzles that coincide spikes in the data.
    #   1.1  locate the spikes
    #   1.2. find puzzle names that correspond to the spikes
    #   1.3. add the puzzle names as an extra column in the dataframe
    # 2. Look at the k puzzles that were added on each iteration

    if flag == '':
        x = np.asarray(idxs_new_puzzles)
    list_values = np.asarray(list_values)

    data = {'Idxs New Puzzles': x,
            'Values': list_values
            }
    df = pd.DataFrame(data, columns=['Idxs New Puzzles', 'Values'])

    fig = px.scatter(df, x='Idxs New Puzzles', y="Values", title=title_name, labels={
        'Idxs New Puzzles': x_label,
        'Values': ""
    })

    filename_to_save_fig = filename_to_save_fig + ".html"
    fig.write_html(filename_to_save_fig, include_mathjax='cdn', include_plotlyjs='cdn')
    return
    # plt.figure ()
    # plt.scatter (x, list_values, s=4, alpha=1.0, color='b') #, edgecolors='black')
    #
    # plt.grid (True)
    # plt.title(title_name)
    # plt.xlabel(x_label)
    # if y_lim_upper is not None and y_lim_lower is not None:
    #     plt.ylim(ymin=y_lim_lower, ymax=y_lim_upper)
    # if x_max is not None and x_min is not None:
    #     plt.xlim(xmin=x_min, xmax=x_max)
    #
    # # plt.show()
    # plt.savefig (filename_to_save_fig)
    # plt.close ()


def get_witness_ordering(flat_list, d):
    witness_ord = []
    for tup in flat_list:
        if "wit" in tup[0]:
            witness_ord.append(tup)
    d[file] = witness_ord
    return d


def separate_names_and_vals(flat_list):
    list_names = []
    list_vals = []
    for tup in flat_list:
        name = tup[0]
        val = tup[1]
        list_names += [name]
        list_vals += [val]
    return list_names, list_vals


def map_witness_puzzles_to_dims(name):
    if (name == "witness_1") or (name == "witness_2"):
        return "1x2"
    elif name == "witness_3":
        return "1x3"
    elif name == "witness_4":
        return "2x2"
    elif (name == "witness_5") or (name == "witness_6"):
        return "3x3"
    elif (name == "witness_7") or (name == "witness_8") or (name == "witness_9"):
        return "4x4"


def find_change_new_metric(new_metric_vals, idx_object):
    list_changes = []
    l_above_0 = []
    idx_above_0 = []
    num_puzzles_in_each_batch = []
    prev_val = 0
    prev_idx = 0

    above_0_per_puzzle = []
    for i in range(len(new_metric_vals)):
        num_p = idx_object[i] - prev_idx
        num_puzzles_in_each_batch += [num_p]
        prev_idx = idx_object[i]

        above_0_per_puzzle += [new_metric_vals[i] / num_p]

        change = new_metric_vals[i] - prev_val
        # print("new_metric_vals[i]", new_metric_vals[i], " prev", prev_val)
        if change > 0.0:
            l_above_0 += [change]
            idx_above_0 += [i]
        prev_val = new_metric_vals[i]
        list_changes += [change]

    print("above_0_per_puzzle", above_0_per_puzzle)
    print(len(new_metric_vals))
    print("idx_above_0", idx_above_0)
    print("")
    print("l_above_0", l_above_0)
    print("")
    print("idxs_solved_batches", idx_object)
    print("")
    print("num_puzzles_in_each_batch", num_puzzles_in_each_batch)
    # assert False
    return list_changes, l_above_0, idx_above_0, num_puzzles_in_each_batch


def find_puzzles_above_0(l_above_0, idx_above_0, num_puzzles_per_batch, flattened_rank_obj, new_metric_vals,
                         idxs_solved_batches,
                         modality='NewMetric'):
    print("num_puzzles_per_batch", num_puzzles_per_batch)

    images_subpath = os.path.join(puzzles_path, modality)
    if not os.path.exists(images_subpath):
        os.makedirs(images_subpath, exist_ok=True)

    l = []
    start_idx = 0
    for i, num in enumerate(num_puzzles_in_each_batch):
        if start_idx == 0:
            end = num
        else:
            end = num + start_idx
        l.append(flattened_rank_obj[start_idx:end])
        start_idx = end

    flat_temp_list = flatten_list(l)
    print("idx_above_0", idx_above_0)
    print("len(flat_temp_list)", len(flat_temp_list))
    sublists_above0 = [flat_temp_list[i] for i in idx_above_0]  # T = [L[i] for i in Idx]
    print(len(sublists_above0))
    assert False
    return sublists_above0


def plot_data_from_DF(df, flag=''): # list_values, idxs_new_puzzles, title_name, filename_to_save_fig, x_label="Number of New Puzzles Solved",
              #y_lim_upper=None, y_lim_lower=None, x_max=None, x_min=None, flag=''):
    #TODO: 1. find puzzles that coincide spikes in the data.
    #   1.1  locate the spikes
    #   1.2. find puzzle names that correspond to the spikes
    #   1.3. add the puzzle names as an extra column in the dataframe
    # 2. Look at the k puzzles that were added on each iteration


    if flag == '':
        x = np.asarray(idxs_new_puzzles)
    list_values = np.asarray(list_values)

    data = {'Idxs New Puzzles': x,
            'Values': list_values
            }
    df = pd.DataFrame(data, columns=['Idxs New Puzzles', 'Values'])

    if flag == "Iterations":
    x_label = flag

    fig = px.scatter(df, x='Idxs New Puzzles', y="Values", title=title_name, labels={
        'Idxs New Puzzles': x_label,
        'Values': ""
    })

    filename_to_save_fig = filename_to_save_fig + ".html"
    fig.write_html(filename_to_save_fig, include_mathjax='cdn', include_plotlyjs='cdn')
    return


class Make_Plots:
    def __init__(self, suffix, num_total_iterations, flag):  # plots_path, puzzles_path,
        self.suffix = suffix
        self.num_total_iterations = num_total_iterations
        self.flag = flag

        self.idxs_solved_batches = \
        open_pickle_file("puzzles_4x4_" + suffix + "/Idxs_rank_data_BFS_" + suffix + "_4x4.pkl")[0]
        self.puzzles_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "puzzles_4x4_" + suffix + "/puzzle_imgs_Batches")
        if not os.path.exists(self.puzzles_path):
            os.makedirs(self.puzzles_path, exist_ok=True)

        self.plots_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "puzzles_4x4_" + suffix + "/plots_Batches")
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path, exist_ok=True)

        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'puzzles_4x4_' + suffix + '/')
        assert os.path.isdir(self.results_path)
        print("number of algo iterations where we solve at least 1 new puzzle", len(self.idxs_solved_batches))
        print("puzzle images path =", self.puzzles_path)
        print("plots_path =", self.plots_path)

    def count_number_new_solved_puzzles(self):
        self.num_New_Puzzles_Solved = []
        x_prev = 0
        for x in self.idxs_solved_batches:
            diff = x - x_prev
            self.num_New_Puzzles_Solved += [diff]
            x_prev = x

        print("num_New_Puzzles_Solved in each iteration", self.num_New_Puzzles_Solved)
        assert sum(self.num_New_Puzzles_Solved) == 2369

    def get_xlabel_and_idxs(self, object):
        if self.flag == "_iterations":
            self.idxs_solved_batches = np.arange(len(object))
            # print(self.idxs_solved_batches)
            self._x_label = 'Iterations'
        else:
            self._x_label = 'Number of New Puzzles Solved'

    def get_title_and_limits(self, file):
        if "New_metric" in file:
            self._title_name = "(grad_c(P) * (theta_n - theta_t))/ ||theta_n - theta_t||"
            self._y_lim_upper = 0.4
            self._y_lim_lower = -0.05

        elif "Cosine" in file:
            self._title_name = "cosine(angle(grad_c(P), (theta_n - theta_t)))"
            self._y_lim_upper = 0.125
            self._y_lim_lower = -0.025

        elif "Dot_Prod" in file:
            self._title_name = "grad_c(P) * (theta_n - theta_t)"
            self._y_lim_upper = 20.0
            self._y_lim_lower = -2.0

        elif "Levin_Cost" in file:
            self._title_name = "Log Levin Cost"
            self._y_lim_upper = None
            self._y_lim_lower = None

        elif "Average_Levin_Cost" in file:
            self._title_name = "Average Log Levin Cost"
            self._y_lim_upper = None
            self._y_lim_lower = None

        elif "Training_Loss" in file:
            self._title_name = "Training Loss (Mean Cross Entropy Loss)"
            self._y_lim_upper = None
            self._y_lim_lower = None

    def make_dataframe(self):
        # -------- OPTIONAL ---------------------------
        cols = [self.idxs_solved_batches]
        col_names = ['Idxs New Puzzles']
        # gather column info and column name
        for file in os.listdir(self.results_path):
            if "_over_P" in file:
                full_filename = os.path.join(self.results_path, file)
                object = open_pickle_file(full_filename)[0]  # open data
                cols.append(object)
                col_names += [file.split("_over_P")[0]]
        # create data frame:
        df = pd.DataFrame(columns=col_names)
        for i in range(len(col_names)):
            this_column = df.columns[i]
            df[this_column] = cols[i]

        df.reset_index(inplace=True)
        df = df.rename(columns={'index': 'Iterations'})  # 'Number of New Puzzles Solved'
        print(df)
        print(df.columns)
        assert False
        # --------------------------------------------

    def walk_through_files(self):
        for i, file in enumerate(os.listdir(self.results_path)):
            prefix = file.split("_4x4.pkl")[0]
            if "Idxs" in file or ".py" in file:
                continue

            if "_over_P" in file:
                full_filename = os.path.join(self.results_path, file)  # 'puzzles_4x4_' + suffix + '/' + file
                print("full_filename", full_filename)
                # open results
                object = open_pickle_file(full_filename)[0]
                # plot data
                self.get_xlabel_and_idxs(object)  # overwrites self.idxs_solved_batches
                self.get_title_and_limits(file)
                plots_filename = os.path.join(self.plots_path, prefix + self.flag)
                print("plots_filename", plots_filename)
                plot_data(object, self.idxs_solved_batches, self._title_name, plots_filename,
                          x_label=self._x_label, y_lim_upper=self._y_lim_upper, y_lim_lower=self._y_lim_lower)


global witness_puzzle
witness_puzzle = WitnessPuzzle()
make_plot = Make_Plots(suffix="theta_n-theta_i", num_total_iterations=93, flag="_iterations")
make_plot.count_number_new_solved_puzzles()
make_plot.walk_through_files()
# the user enters: -----
# num_total_iterations = 93, flag = "_iterations"
