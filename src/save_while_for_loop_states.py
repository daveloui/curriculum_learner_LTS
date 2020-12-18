import numpy as np
import tensorflow as tf
import os
from os.path import join
import pickle
from tensorflow import keras
from tensorflow.keras import layers

from models.model_wrapper import KerasManager, KerasModel


def save_while_loop_state(puzzle_dims, iteration, total_expanded, total_generated, budget, current_solved_puzzles,
                          last_puzzle, start, start_while):

    checkpoint_folder = 'checkpoint_data_' + str(puzzle_dims) + "/"
    if not os.path.exists (checkpoint_folder):
        os.makedirs (checkpoint_folder, exist_ok=True)
    filename = os.path.join (checkpoint_folder, 'While_Loop_State.pkl')

    d_wl = {}
    d_wl["iteration"] = iteration
    d_wl["total_expanded"] = total_expanded
    d_wl["total_generated"] = total_generated
    d_wl["budget"] = budget
    d_wl["current_solved_puzzles"] = current_solved_puzzles
    d_wl["last_puzzle"] = last_puzzle
    d_wl["start"] = start
    d_wl["start_while"] = start_while

    outfile = open (filename, 'wb')
    pickle.dump (d_wl, outfile)
    outfile.close ()


def restore_while_loop_state(puzzle_dims):
    print("inside restore_while_loop_state")
    checkpoint_folder = 'checkpoint_data_' + str(puzzle_dims) + "/"
    assert os.path.exists (checkpoint_folder)
    filename = os.path.join (checkpoint_folder, 'While_Loop_State.pkl')

    with open (filename, 'rb') as infile:
        d_wl = pickle.load (infile)
    infile.close ()

    iteration = d_wl["iteration"]
    total_expanded = d_wl["total_expanded"]
    total_generated = d_wl["total_generated"]
    budget = d_wl["budget"]
    current_solved_puzzles = d_wl["current_solved_puzzles"]
    last_puzzle = d_wl["last_puzzle"]
    start = d_wl["start"]
    start_while = d_wl["start_while"]

    return iteration, total_expanded, total_generated, budget, current_solved_puzzles, last_puzzle, start, start_while


def save_for_loop_state(n_while, for_loop_index, number_solved, at_least_one_got_solved, iteration, total_expanded,
                         total_generated, budget, current_solved_puzzles, last_puzzle, ordering, puzzle_dims):
    d_fl = {}
    d_fl["n_while"] = n_while
    d["for_loop_index"] = for_loop_index
    d["number_solved"] = number_solved
    d["at_least_one_got_solved"] = at_least_one_got_solved
    d["iteration"] = iteration

    d["total_expanded"] = total_expanded
    d["total_generated"] = total_generated
    d["budget"] = budget
    d["current_solved_puzzles"] = current_solved_puzzles
    d["last_puzzle"] = last_puzzle

    d["ordering"] = ordering
    d["puzzle_dims"] = puzzle_dims

    checkpoint_folder = 'checkpoint_data_' + str(puzzle_dims) + "/"
    if not os.path.exists (checkpoint_folder):
        os.makedirs (checkpoint_folder, exist_ok=True)
    filename = 'For_Loop_State'
    filename = os.path.join (checkpoint_folder, filename + '.pkl')
    print ("len(d_fl) =", len (d_fl))
    outfile = open (filename, 'wb')
    pickle.dump (d_fl, outfile)
    outfile.close ()

