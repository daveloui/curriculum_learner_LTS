import pickle

import numpy as np
import matplotlib.pyplot as plt


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
print("aver_levin_cost_data")
object = open_pickle_file("logs_large/4x4_grouped_P/aver_levin_cost_data_4x4-Witness-CrossEntropyLoss.pkl")
# print(type(object))
print(len(object))
print(object)

print("")
print("cosine_data")
object = open_pickle_file("logs_large/4x4_grouped_P/cosine_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(len(object))
print(object)

print("")
print("dot_prod_data")
object = open_pickle_file("logs_large/4x4_grouped_P/dot_prod_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(object)


print("")
print("levin_cost_data")
object = open_pickle_file("logs_large/4x4_grouped_P/levin_cost_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(object)

print("")
print("training_loss_data")
object = open_pickle_file("logs_large/4x4_grouped_P/training_loss_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(object)



def plot_data(list_values, idxs_new_puzzles, title_name, filename_to_save_fig, special_x, special_y,
              x_label="Number of Puzzles Solved", y_lim_upper=None, y_lim_lower=None, x_max=2400, x_min=-5):
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


# d = {}
# plots_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4_grouped_P/plots")
# print ("plots_path =", plots_path)
# if not os.path.exists (plots_path):
#     os.makedirs (plots_path, exist_ok=True)
#
# for file in os.listdir('puzzles_4x4/'):
#     if "Rank_" in file:
#
#         full_filename = 'puzzles_4x4/' + file
#         object = open_pickle_file(full_filename)
#         print("len(obj)", len(object))
#         flat = flatten_list(flatten_list(object))
#         assert len (flat) == 2369
#         list_names, list_vals = separate_names_and_vals(flat)
#
#         special_x, special_y = find_special_vals (object)
#         d = get_witness_ordering(flat, d)
#
#         for i, sub_sublist in enumerate(object[0]):
#             print("len(sub_sublist) =", len(sub_sublist))
#             for tuple_el in sub_sublist:
#                 p_name = tuple_el[0]
#                 if "wit" in p_name:
#                     print("pname ", p_name, " found in sublist ", i)
#             # print("")
#         print("")
#
#         continue  # TODO: debug -- uncomment
#
#         if "DotProd" in file:
#             title_name = "grad_c(p) * (theta_n - theta_i)"
#             # sublist = d[file]
#         elif "Cosine" in file:
#             title_name = "cosine(angle(grad_c(p), (theta_n - theta_i)))"
#         else:
#             title_name = "Levin Cost"
#
#         plots_filename = os.path.join(plots_path, file.split("_BFS_4x4.pkl")[0])
#         print("plots_filename", plots_filename)
#
#         plot_data(list_vals, idx_object[0], title_name, plots_filename, special_x, special_y)
#     # continue # TODO: debug -- uncomment
#
#     elif "Ordering" in file:
#         filename = file.split("_BFS")[0]
#         print ("filename", filename)
#         full_filename = 'puzzles_4x4/' + file
#         object = open_pickle_file(full_filename)
#         print("len(ordering object)", len(object))
#         print(object)
#
# print("")
# print("d", d)
# print("")