import numpy as np
import sys
import pickle
from six.moves import cPickle
import os
from os.path import isfile, join
import sys
import time

np.random.seed (1)


class Trajectory ():
    def __init__(self, states, actions, solution_costs, expanded, solution_pi=0.0):
        self._states = states
        self._actions = actions
        self._expanded = expanded
        self._non_normalized_expanded = expanded
        self._solution_costs = solution_costs
        self._solution_pi = solution_pi
        self._is_normalized = False

    def get_states(self):
        return self._states

    def get_actions(self):
        return self._actions

    def get_expanded(self):
        return self._expanded

    def get_solution_costs(self):
        return self._solution_costs

    def get_non_normalized_expanded(self):
        return self._non_normalized_expanded

    def get_solution_pi(self):
        return self._solution_pi

    def normalize_expanded(self, factor):
        if not self._is_normalized:
            self._expanded /= factor
            self._is_normalized = True


class TrajectoryV2 ():
    def __init__(self, states, actions):  # , solution_costs, solution_pi=0.0):
        self.states = states
        self.actions = actions
        self.states_data = []
        # self._solution_costs = solution_costs
        # self._solution_pi = solution_pi
        # self._is_normalized = False

    def process_states(self):
        self.states_data = [s.convert_2_dict_for_image_acquisition () for s in self.states]  # a list of dictionaries
        assert len (self.states) == len (self.states_data)

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions


class Memory ():
    def __init__(self):
        self._trajectories = []
        self._max_expanded = -sys.maxsize
        self._puzzle_names = []

    def add_trajectory(self, trajectory, puzzle_name):
        if trajectory.get_expanded () > self._max_expanded:
            self._max_expanded = trajectory.get_expanded ()
        self._trajectories.append (trajectory)

        if puzzle_name not in self._puzzle_names:
            self._puzzle_names += [puzzle_name]

    def shuffle_trajectories(self):
        self._random_indices = np.random.permutation (len (self._trajectories))

    def next_trajectory(self):

        for i in range (len (self._trajectories)):
            traject = np.array (self._trajectories)[self._random_indices[i]]
            traject.normalize_expanded (self._max_expanded)
            yield traject

    def number_trajectories(self):
        return len (self._trajectories)

    def merge_trajectories(self, other):
        for t in other._trajectories:
            self._trajectories.append (t)

    def clear(self):
        self._trajectories.clear ()
        self._max_expanded = -sys.maxsize
        self._puzzle_names = []


class MemoryV2 ():
    '''
    This class contains two member variables:
    self.position: the position in time in which a puzzle was solved
    self.dict: a dictionary whose keys are = self.position,
            whose values are a list containing the trajectory data (data on states and actions)
    '''
    def __init__(self, path_to_save_data, puzzle_dims, how_do_we_add_puzzle_solutions):
        self.position = 1  # GET RID OF THIS
        self.dict = {}
        self.memory_version = 'v2'
        self.path_to_save_data = path_to_save_data
        self.puzzle_dims = puzzle_dims
        self.dict_images_actions = {}

        # added:
        self.trajectories_dict = {}

        if how_do_we_add_puzzle_solutions == "add_new_solutions_to_old_puzzles":
            self.save_mode = 'wb'
        else:
            self.save_mode = 'ab'

    def add_trajectory(self, trajectory_obj, puzzle_name):
        new_trajectory_obj = TrajectoryV2 (trajectory_obj._states, trajectory_obj._actions)
        new_trajectory_obj.process_states ()
        self.dict[puzzle_name] = [self.position, new_trajectory_obj.states_data, trajectory_obj._actions]
        self.position += 1
        self.trajectories_dict[puzzle_name] = trajectory_obj

    def number_trajectories(self):
        return len (self.trajectories_dict)

    def save_trajectories_dict(self):
        new_dict = {}
        for k, v in self.trajectories_dict.items():
            states = v.get_states()
            actions = v.get_actions()
            new_dict[k] = [states, actions]

    def save_data(self):
        start = time.time ()
        filename = 'BFS_memory_' + self.puzzle_dims + '.pkl'
        filename = os.path.join (self.path_to_save_data, filename)

        with open (filename, self.save_mode) as fout:
            cPickle.dump (self.dict, fout, protocol=cPickle.HIGHEST_PROTOCOL)
        fout.close ()
        # print("finished saving data")
        end = time.time ()
        print ("time to save BFS_memory", end - start)

    def store_puzzle_images_actions(self, puzzle_name, images_p_array, actions_p_array):
        self.dict_images_actions[puzzle_name] = [images_p_array, actions_p_array]

    def retrieve_puzzle_images(self, puzzle_name):
        return self.dict_images_actions[puzzle_name][0]

    def retrieve_labels(self, puzzle_name):
        return self.dict_images_actions[puzzle_name][1]

    def clear(self):
        self.dict = {}
        self.dict_images_actions = {}
        self.trajectories_dict = {}