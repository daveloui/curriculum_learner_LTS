import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


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


def plot_data_html(list_values, idxs_new_puzzles, title_name, filename_to_save_fig,
                   x_label="Number of New Puzzles Solved", flag=''):
    if flag == '':
        x = np.asarray(idxs_new_puzzles)
    data = {'Idxs New Puzzles': x,
            'Values': np.asarray(list_values)
            }
    df = pd.DataFrame(data, columns=['Idxs New Puzzles', 'Values'])
    fig = px.scatter(df, x='Idxs New Puzzles', y="Values", title=title_name,
                     labels={'Idxs New Puzzles': x_label, 'Values': ""})

    filename_to_save_fig = filename_to_save_fig + ".html"
    fig.write_html(filename_to_save_fig, include_mathjax='cdn', include_plotlyjs='cdn')
    return


def plot_data_mpl(list_values, idxs_new_puzzles, title_name, filename_to_save_fig,
                  x_label="Number of New Puzzles Solved", y_lim_upper=None, y_lim_lower=None, x_max=None,
                  x_min=None, flag=''):
    if flag == '':
        x = np.asarray(idxs_new_puzzles)
    plt.figure ()
    plt.scatter (x, np.asarray(list_values), s=4, alpha=1.0, color='b')
    plt.grid (True)
    plt.title(title_name)
    plt.xlabel(x_label)
    if y_lim_upper is not None and y_lim_lower is not None:
        plt.ylim(ymin=y_lim_lower, ymax=y_lim_upper)
    if x_max is not None and x_min is not None:
        plt.xlim(xmin=x_min, xmax=x_max)
    # plt.show()
    plt.savefig (filename_to_save_fig + ".png")
    plt.close ()


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

    def walk_through_files(self):
        for i, file in enumerate(os.listdir(self.results_path)):
            prefix = file.split("_4x4.pkl")[0]
            if "Idxs" in file or ".py" in file:
                continue

            if "_over_P" in file:
                full_filename = os.path.join(self.results_path, file)  # 'puzzles_4x4_' + suffix + '/' + file
                print("full_filename", full_filename)
                # open results:
                object = open_pickle_file(full_filename)[0]
                # plot data
                self.get_xlabel_and_idxs(object)  # overwrites self.idxs_solved_batches
                self.get_title_and_limits(file)
                plots_filename = os.path.join(self.plots_path, prefix + self.flag)
                print("plots_filename", plots_filename)
                plot_data_html(object, self.idxs_solved_batches, self._title_name, plots_filename,
                               x_label=self._x_label)
                plot_data_mpl(object, self.idxs_solved_batches, self._title_name, plots_filename,
                          x_label=self._x_label, y_lim_upper=self._y_lim_upper, y_lim_lower=self._y_lim_lower)


# the user enters: -----
# num_total_iterations = 93, flag = "_iterations"
make_plot = Make_Plots(suffix="theta_n-theta_i", num_total_iterations=93, flag="_iterations")
make_plot.count_number_new_solved_puzzles()
make_plot.walk_through_files()
