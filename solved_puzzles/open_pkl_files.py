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
from Witness_Puzzle_Image import WitnessPuzzle


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


# Step 1: open object, check size and type
print("Rank_MaxDotProd_BFS_4x4.pkl")
object = open_pickle_file("puzzles_4x4/Rank_MaxDotProd_BFS_4x4.pkl")
# print(type(object))
print(len(object[0]))

# verify total number of puzzles, check that flattened list maintains first and last puzzles
# first_last_puzzles = []
# for sublist in object:
#     len_j = len(sublist)
#     print("sublist level 1 has length =", len_j)
#     # summ = 0
#     prev_len = 0
#     for i, sub_sublist in enumerate(sublist):
#         len_i = len(sub_sublist)
#         print("sub_sublist level 2", len_i)
#         # summ += len_i
#
#         first_i = sub_sublist[0][0]
#         last_i = sub_sublist[-1][0]
#
#         if first_i != last_i:
#             first_last_puzzles.append ([(prev_len + 0, first_i), (prev_len + len_i - 1, last_i)])
#         else:
#             first_last_puzzles.append([(prev_len, first_i)])
#
#         prev_len += len_i
#         for tuple_el in sub_sublist:
#             p_name = tuple_el[0]
#             if "wit" in p_name:
#                 print("pname ", p_name, " found in sublist ", i)
#         print("")
#     print("")
# #     print("summ =", summ)
# # assert summ == 2369
# print("first_last", first_last_puzzles)


def plot_data(list_values, idxs_new_puzzles, title_name, filename_to_save_fig, special_x, special_y,
              x_label="Puzzles Solved", y_lim_upper=None, y_lim_lower=None, x_max=2400, x_min=-5):
    plt.figure ()
    for xc in idxs_new_puzzles:
        plt.axvline (x=xc, c='orange', linestyle='--', alpha=0.6)
    x = np.arange(0, 2369)
    list_values = np.asarray(list_values)
    plt.scatter (x, list_values, s=4, alpha=1.0, color='b')

    plt.grid (True)
    plt.title(title_name)
    plt.xlabel(x_label)
    if y_lim_upper is not None and y_lim_lower is not None:
        plt.ylim(ymin=y_lim_lower, ymax=y_lim_upper)
    if x_max is not None and x_min is not None:
        plt.xlim(xmin=x_min, xmax=x_max)

    # plt.show()
    plt.savefig (filename_to_save_fig)
    # plt.close ()

    string_list = filename_to_save_fig.split("plots/")
    print(title_name)
    if "Levin Cost" in title_name:
        zoomed_title = "Zoomed Levin Cost"
    elif "Cosines" in title_name:
        zoomed_title = "Zoomed " + title_name
    else:
        zoomed_title = "Zoomed " + title_name

    zoomed_filename_to_save_fig = string_list[0] + "plots/" + "Zoom_" + string_list[1]
    print("zoomed_filename_to_save_fig", zoomed_filename_to_save_fig)
    plt.xlim(xmin=2200, xmax=2375)
    plt.title (zoomed_title)
    # plt.show()
    # plt.savefig (zoomed_filename_to_save_fig)
    plt.close()

    return


def get_witness_ordering(flat_list, d):
    witness_ord = []
    for tup in flat_list:
        if "wit" in tup[0]:
            witness_ord.append (tup)
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


def find_special_vals(loaded_object):
    for sublist in loaded_object[0]:
        pass
    return None, None



print("Idxs_rank_data_BFS_4x4.pkl")
idx_object = open_pickle_file("puzzles_4x4/Idxs_rank_data_BFS_4x4.pkl")
print(len(idx_object[0]))
print("")


d = {}
plots_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4/plots")
print ("plots_path =", plots_path)
if not os.path.exists (plots_path):
    os.makedirs (plots_path, exist_ok=True)

for file in os.listdir('puzzles_4x4/'):
    if "Rank_" in file:
        full_filename = 'puzzles_4x4/' + file
        object = open_pickle_file(full_filename)
        flat = flatten_list(flatten_list(object))
        assert len (flat) == 2369
        list_names, list_vals = separate_names_and_vals(flat)

        special_x, special_y = find_special_vals (object)
        d = get_witness_ordering(flat, d)

        if "DotProd" in file:
            title_name = "grad_c(p) * (theta_n - theta_i)"
            # sublist = d[file]
        elif "Cosine" in file:
            title_name = "cosine(angle(grad_c(p), (theta_n - theta_i)))"
        else:
            title_name = "Levin Cost"

        plots_filename = os.path.join(plots_path, file.split("_BFS_4x4.pkl")[0])
        print("plots_filename", plots_filename)

        plot_data(list_vals, idx_object[0], title_name, plots_filename, special_x, special_y)

    elif "Ordering" in file:
        full_filename = 'puzzles_4x4/' + file
        print("file", full_filename)
        object = open_pickle_file(full_filename)
        print("len(ordering object)", len(object))
        print(object)
        print("")

print("")
print("d", d)
print("")


# print("")
# print("Ordering_DotProds_BFS_4x4.pkl")
# object = open_pickle_file("puzzles_4x4/Ordering_DotProds_BFS_4x4.pkl")
# print(type(object))
# print(len(object[0]))
# print(object)
#


#
# # print("")
# # print("Rank_MinLevinCost_BFS_4x4.pkl")
# # object = open_pickle_file("puzzles_4x4/Rank_MinLevinCost_BFS_4x4.pkl")
# # print(type(object))
# # print(len(object[0]))
# # print(object[0][0][0:20])
#
# print("")
# print("Ordering_LevinScores_BFS_4x4.pkl")
# object = open_pickle_file("puzzles_4x4/Ordering_LevinScores_BFS_4x4.pkl")
# print(type(object))
# print(len(object[0]))
# print(object)