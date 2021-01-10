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
# item_0 = object[0][0]
# print(list(object[0].values())[0])
for k, v_list in object[0].items():
    if k == p_name:
        v_list_p = v_list
print("len(v_list_p)", len(v_list_p))
print(v_list_p)


models_folder = 'trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/'
puzzle_dims = '4x4'
ordering_folder = 'solved_puzzles/puzzles_' + puzzle_dims  # + "_debug_data"


# MemoryManager.register ('MemoryModel', MemoryModel)
# with MemoryManager () as mem_manager:
#     memory_model = mem_manager.MemoryModel ()
#     memory_model.initialize (run_idx, ncpus, chunk_size)

KerasManager.register ('KerasModel', KerasModel)
with KerasManager () as manager:
    nn_model = manager.KerasModel ()
    nn_model.initialize ("CrossEntropyLoss", "Levin", two_headed_model=False)
    nn_weights_file = "trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/pretrained_weights_1.h5"
    nn_model.load_weights (nn_weights_file)
    theta_i = nn_model.retrieve_layer_weights ()

    theta_diff = compute_cosines (nn_model, models_folder, 2)
    print(type(theta_diff))

    memory_v2 = MemoryV2 (ordering_folder, puzzle_dims, "only_add_solutions_to_new_puzzles")

    # argmax_p_dot_prods, Rank_dot_prods, argmax_p_cosines, Rank_cosines = compute_rank ([p_name], nn_model,
    #                                                                                    theta_diff,
    #                                                                                    memory_v2,
    #                                                                                    4, 1, 1,
    #                                                                                    False)


# theta_diff = compute_cosines (nn_model, self._models_folder, while_loop_iter)