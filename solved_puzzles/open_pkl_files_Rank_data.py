import os
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import ipdb

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


def plot_data(list_values, idxs_new_puzzles, title_name, filename_to_save_fig, x_label="Number of New Puzzles Solved",
              y_lim_upper=None, y_lim_lower=None, x_max=None, x_min=None, vertical_lines=False):
    plt.figure ()
    if vertical_lines:
        for xc in idxs_new_puzzles:
            plt.axvline (x=xc, c='orange', linestyle='--', alpha=0.6)

    x_values = np.asarray(range(1, len(list_values)+1))  #np.arange(0, 65)
    # x_values = np.arange(len (list_vals))
    list_values = np.asarray(list_values)
    print ("x_values.shape", x_values.shape)
    print("list_values.shape", list_values.shape)

    plt.scatter (x_values, list_values, s=4, alpha=1.0, color='b') #, edgecolors='black')
    plt.grid (True)
    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylim(ymin=y_lim_lower, ymax=y_lim_upper)
    if x_max is not None and x_min is not None:
        plt.xlim(xmin=x_min, xmax=x_max)
    # plt.show()
    plt.savefig (filename_to_save_fig)
    print("filename_to_save_fig", filename_to_save_fig)
    plt.close ()


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


def find_top_k(nested_list, k):
    new_vals = []
    names_dict = {}
    idx_obj = []
    prev = 0
    for i, sublist in enumerate(nested_list):
        first_k = sublist[:k]
        l = len(first_k)
        l += prev
        idx_obj += [l]
        prev = l

        k_names_in_sublist = []
        for tup in first_k:
            name = tup[0]
            val = tup[1]
            k_names_in_sublist += [name]
            new_vals += [val]  # for plotting
        names_dict[i] = k_names_in_sublist  # for later getting the puzzles of the top k of each iteration
    return names_dict, new_vals, idx_obj


# Open ordering of puzzles
ord = open_pickle_file('puzzles_4x4_theta_n-theta_i/Ordering_NewMetric_BFS_theta_n-theta_i_4x4.pkl')[0]
# Open ranking of puzzles
rank = open_pickle_file('puzzles_4x4_theta_n-theta_i/Rank_NewMetric_BFS_theta_n-theta_i_4x4.pkl')[0]
# Open new metric
new_metric = open_pickle_file('puzzles_4x4_theta_n-theta_i/New_metric_over_P_theta_n-theta_i_4x4.pkl')[0]


suffix = "theta_n-theta_i"
idx_object = open_pickle_file("puzzles_4x4_" + suffix + "/Idxs_rank_data_BFS_" + suffix + "_4x4.pkl")

num_New_Puzzles_Solved = []
x_prev = 0
for x in idx_object[0]:
    diff = x - x_prev
    num_New_Puzzles_Solved += [diff]
    x_prev = x
while_loop_iter = len(num_New_Puzzles_Solved)

# create folders:
puzzles_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4_" + suffix + "/puzzle_imgs")
if not os.path.exists (puzzles_path):
    os.makedirs (puzzles_path, exist_ok=True)

plots_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4_" + suffix + "/plots")
if not os.path.exists (plots_path):
    os.makedirs (plots_path, exist_ok=True)


witness_puzzle = WitnessPuzzle ()
d = {}
for file in os.listdir('puzzles_4x4_' + suffix + '/'):
    print("")
    if "Idxs" in file or ".py" in file:
        continue

    if "Rank_" in file:  # or "Ordering_" in file:
        full_filename = 'puzzles_4x4_' + suffix + '/' + file
        object = open_pickle_file(full_filename)
        print ("filename", file)
        print("len(object[0])", len(object[0]))
        print ("")
        flat = flatten_list(flatten_list(object))
        assert len (flat) == 2369

        # get the ordering of witness puzzles only:
        d = get_witness_ordering (flat, d)
        WP_ordering_path = os.path.join ("puzzles_4x4_" + suffix + "/Witness_Puzzles_Ordering")
        if not os.path.exists (WP_ordering_path):
            os.makedirs (WP_ordering_path, exist_ok=True)
        WP_dict_file = join (WP_ordering_path, file.split("_BFS_")[0])

        w = csv.writer (open (WP_dict_file, "w"))
        for key, val in d.items ():
            w.writerow ([key, val])
        d = {}
        k = 5
        flag = '_top_k'  # or ''
        vertical_lines = False
        y_min = None
        y_max = None
        if "DotProd" in file:
            title_name = "grad_c(p) * (theta_{t+1} - theta_t)"
        elif "Cosine" in file:
            title_name = "cosine(angle(grad_c(p), (theta_{t+1} - theta_t)))"
        elif "NewMetric" in file:
            title_name = "grad_c(p) * (theta_{t+1} - theta_t) / ||(theta_{t+1} - theta_t)||"
        else:
            title_name = "Log Levin Cost"

        if flag == '_top_k':
            nested_list = flatten_list (object)  # each sublist corresponds to one iteration where at least one new puzzle gets solved
            dict_names, list_vals, idxs_new_puzzles = find_top_k (nested_list, k)
            x_values = np.arange(len (list_vals))
            x_label = 'Number of New Puzzles Solved (top ' + str(k) + ')'

            Top_k_path = os.path.join ("puzzles_4x4_" + suffix + "/Top_" + str(k) + "_Ordering")
            if not os.path.exists (Top_k_path):
                os.makedirs (Top_k_path, exist_ok=True)
            Top_k_dict_file = join (Top_k_path, file.split ("_BFS_")[0])

            # save as txt file
            w = csv.writer (open (Top_k_dict_file, "w"))
            for key, val in dict_names.items ():
                w.writerow ([key, val])
            # save copy as pkl file
            output = open (Top_k_dict_file + '.pkl', 'wb')
            pickle.dump (dict_names, output)
            output.close ()
        else:
            list_names, list_vals = separate_names_and_vals(flat)
            x_values = np.arange(len (list_vals))   #np.asarray(idx_object[0])
            x_label = 'Number of New Puzzles Solved'
            idxs_new_puzzles = idx_object[0]

        plots_filename = os.path.join(plots_path, file.split("_4x4.pkl")[0] + flag + "_vertical_lines_" +
                                      str(vertical_lines))  #os.path.join(plots_path, "Zoomed_" + file.split("_BFS_4x4.pkl")[0])
        plot_data(list_vals, idxs_new_puzzles, title_name, plots_filename, x_label=x_label, y_lim_upper=y_max,
                  y_lim_lower=y_min, vertical_lines=vertical_lines)
