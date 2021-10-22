import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from os.path import join
from concurrent.futures.process import ProcessPoolExecutor
import math
import numpy as np
import tensorflow as tf
import random
import ipdb

random.seed (1)
np.random.seed (1)
tf.random.set_seed (1)

from models.memory import Memory, MemoryV2


class ProblemNode:
    def __init__(self, k, n, name, instance):
        self._k = k
        self._n = n
        self._name = name
        self._instance = instance
        self._cost = 125 * (4 ** (self._k - 1)) * self._n * self._k * (self._k + 1)

    def __lt__(self, other):
        """
        Function less-than used by the heap
        """
        if self._cost != other._cost:
            return self._cost < other._cost
        else:
            return self._k < other._k

    def get_budget(self):
        budget = 125 * (4 ** (self._k - 1)) * self._n - (125 * (4 ** (self._k - 1) * (self._n - 1)))
        return budget

    def get_name(self):
        return self._name

    def get_k(self):
        return self._k

    def get_n(self):
        return self._n

    def get_instance(self):
        return self._instance


class Bootstrap_No_Debug:
    def __init__(self, states, output, ncpus=1, initial_budget=1, gradient_steps=10,
                 k_expansions=1):
        self._states = states
        self._model_name = output
        self._number_problems = len (states)
        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps
        self._batch_size = 32
        self._all_puzzle_names = set (states.keys ())  ## FD what do we use this for?
        self._puzzle_dims = self._model_name.split ('-')[0]  # self._model_name has the form
        self._log_folder = 'logs_large/' + self._puzzle_dims
        self._models_folder = 'trained_models_large/BreadthFS_' + self._model_name
        self._ordering_folder = 'solved_puzzles/puzzles_' + self._puzzle_dims

        if not os.path.exists (self._models_folder):
            os.makedirs (self._models_folder, exist_ok=True)

        if not os.path.exists (self._log_folder):
            os.makedirs (self._log_folder, exist_ok=True)

        if not os.path.exists (self._ordering_folder):
            os.makedirs (self._ordering_folder, exist_ok=True)

        self._cosine_data = []
        self._dot_prod_data = []
        self._levin_costs = []
        self._average_levin_costs = []
        self._training_losses = []

    def map_function(self, data):
        gbs = data[0]
        nn_model = data[1]
        max_steps = data[2]

        return gbs.solve (nn_model, max_steps)

    def _solve_uniform_online(self, planner, nn_model, parameters):
        parallelize_with_NN = bool (int (parameters.parallelize_with_NN))
        memory = Memory ()
        memory_v2 = MemoryV2 (self._ordering_folder, self._puzzle_dims, "only_add_solutions_to_new_puzzles")

        iteration = 1
        total_expanded = 0
        total_generated = 0
        budget = self._initial_budget
        start = time.time ()
        current_solved_puzzles = set ()
        last_puzzle = list (self._states)[-1]  # self._states is a dictionary of puzzle_file_name, puzzle
        start_while = time.time ()

        while_loop_iter = 1
        while len (current_solved_puzzles) < self._number_problems:
            number_solved = 0
            batch_problems = {}
            s_for_loop = time.time ()
            print("inside while loop")

            j = 0
            for name, state in self._states.items ():  # iterate through all the puzzles, try to solve however many you have with a current budget
                batch_problems[name] = state
                print("inside for loop")

                if len (batch_problems) < self._batch_size and last_puzzle != name:
                    print("here")
                    continue  # we only proceed if the number of elements in batch_problems == self._batch_size

                print("now here")
                j += 1
                num_states_still_to_try_to_solve = self._number_problems - (j * self._batch_size)

                # once we have self._batch_size puzzles in batch_problems, we look for their solutions and train NN
                if parallelize_with_NN:
                    chunk_size_heuristic = math.ceil (len (batch_problems) / (self._ncpus * 4))
                    with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                        args = ((state, name, budget, nn_model) for name, state in batch_problems.items ())
                        results = executor.map (planner.search_for_learning, args, chunksize=chunk_size_heuristic)

                    for result in results:
                        has_found_solution = result[0]
                        trajectory = result[1]  # the solution trajectory
                        total_expanded += result[2]
                        total_generated += result[3]
                        puzzle_name = result[4]

                        if has_found_solution:
                            memory.add_trajectory (trajectory, name)

                        if has_found_solution and puzzle_name not in current_solved_puzzles:
                            number_solved += 1
                            current_solved_puzzles.add (puzzle_name)
                            memory_v2.add_trajectory (trajectory, puzzle_name)

                    print ("")
                    print ("time spent on for loop so far = ", time.time () - s_for_loop)
                    print ("num puzzles we still need to try to solve =",
                           num_states_still_to_try_to_solve)
                    print ("number of puzzles we have tried to solve already", (j * self._batch_size))
                    print ("puzzles solved so far (in total, over all while loop/for loop iterations)",
                           len (current_solved_puzzles))
                    print ("number of puzzles solved in the current while loop iteration =", number_solved)
                    print ("while_loop_iter =", while_loop_iter, " iteration =", iteration)
                    print ("")
                else:
                    s_solve_puzzles = time.time ()
                    for name, state in batch_problems.items ():
                        args = (state, name, budget, nn_model)
                        has_found_solution, trajectory, total_expanded, total_generated, puzzle_name = \
                            planner.search_for_learning (args)
                        if has_found_solution:
                            memory.add_trajectory (trajectory, name)

                        if has_found_solution and puzzle_name not in current_solved_puzzles:
                            number_solved += 1
                            current_solved_puzzles.add (puzzle_name)
                            memory_v2.add_trajectory (trajectory, puzzle_name)
                    e_solve_puzzles = time.time ()
                    print ("time to solve batch of ", self._batch_size, "= ", e_solve_puzzles - s_solve_puzzles)

                batch_problems.clear ()

            if memory.number_trajectories () > 0:
                print("NOW TRAINING -----------------")
                for _ in range (self._gradient_steps):
                    loss = nn_model.train_with_memory (memory)
                    print ('Loss: ', loss)
                memory.clear ()
                nn_model.save_weights (
                    join (self._models_folder, "pretrained_weights_" + str (while_loop_iter) + ".h5"))
                while_loop_iter += 1
                print ("finished training -----------")
            end = time.time ()

            with open (join (self._log_folder, 'training_bootstrap_' + self._model_name), 'a') as results_file:
                results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (while_loop_iter,
                                                                                               iteration,
                                                                                               number_solved,
                                                                                               self._number_problems - len (current_solved_puzzles),
                                                                                               budget,
                                                                                               total_expanded,
                                                                                               total_generated,
                                                                                               end - start)))
                results_file.write ('\n')

            print ('Number solved: ', number_solved, ' Number still to solve', self._number_problems -
                   len (current_solved_puzzles))
            if number_solved == 0:
                budget *= 2
                print ('Budget: ', budget)
                continue

            unsolved_puzzles = self._all_puzzle_names.difference (current_solved_puzzles)
            with open (join (self._log_folder, 'unsolved_puzzles_' + self._model_name), 'a') as file:
                for puzzle in unsolved_puzzles:  # current_solved_puzzles:
                    file.write ("%s," % puzzle)
                file.write ('\n')
                file.write ('\n')

            iteration += 1
            end_while = time.time()
            print ("time for while-loop iter =", end_while - start_while)
            print ("")

        if len (current_solved_puzzles) == self._number_problems:
            nn_model.save_weights (join (self._models_folder,
                                         "Final_weights_NoDebug.h5"))
            memory_v2.save_data ()

    def solve_problems(self, planner, nn_model, parameters):
        self._solve_uniform_online (planner, nn_model, parameters)
