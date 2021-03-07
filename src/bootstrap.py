import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from os.path import join
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math
import numpy as np
import tensorflow as tf
import random

random.seed (1)
np.random.seed (1)
tf.random.set_seed (1)


from models.memory import Memory, MemoryV2
from compute_cosines import retrieve_batch_data_solved_puzzles, check_if_data_saved, retrieve_all_batch_images_actions, \
    compute_cosines, save_data_to_disk, find_argmax, compute_levin_cost, find_minimum, compute_rank, compute_rank_mins, \
    compute_sum_weights
from save_while_for_loop_states import save_while_loop_state, restore_while_loop_state


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


class Bootstrap:
    def __init__(self, states, output, scheduler, use_GPUs=False, ncpus=1, initial_budget=1, gradient_steps=10):
        # parallelize_with_NN=True):

        self._states = states
        self._model_name = output
        self._number_problems = len (states)

        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps

        self._batch_size = 32
        self._kmax = 10
        self._scheduler = scheduler
        # self._parallelize_with_NN = parallelize_with_NN

        self._all_puzzle_names = set (states.keys ())  ## FD what do we use this for?
        self._puzzle_dims = self._model_name.split ('-')[0]  # self._model_name has the form '<puzzle dimension>-<problem domain>-<loss name>'

        self._log_folder = 'logs_large/' + self._puzzle_dims  # + "_debug_data"
        self._models_folder = 'trained_models_large/BreadthFS_' + self._model_name  # + "_debug_data"
        self._ordering_folder = 'solved_puzzles/puzzles_' + self._puzzle_dims  # + "_debug_data"

        if not os.path.exists (self._models_folder):
            os.makedirs (self._models_folder, exist_ok=True)

        if not os.path.exists (self._log_folder):
            os.makedirs (self._log_folder, exist_ok=True)

        if not os.path.exists (self._ordering_folder):
            os.makedirs (self._ordering_folder, exist_ok=True)

        self._new_metric_data_P = []
        self._cosine_data_P = []
        self._dot_prod_data_P = []
        self._levin_costs_P = []
        self._average_levin_costs_P = []
        self._training_losses_P = []
        self._grads_l2_P = []
        self.use_GPUs = use_GPUs

    def map_function(self, data):
        gbs = data[0]
        nn_model = data[1]
        max_steps = data[2]

        return gbs.solve (nn_model, max_steps)

    def initialize(self, parameters):
        print ("inside initialize")
        self.parallelize_with_NN = bool (int (parameters.parallelize_with_NN))
        print ("parallelize with NN? ", self.parallelize_with_NN)
        print ("")
        # TODO:
        # step 1: check if we have already stored the solution trajectories and images of all the puzzles
        # already_saved_data = check_if_data_saved (self._puzzle_dims) # -- checks if we already created the file
        # 'Solved_Puzzles_Batch_Data' and saved in the folder: batch_folder = 'solved_puzzles/' + puzzle_dims + "/"
        self.ordering_new_metric = []
        self.Rank_max_new_metric = []
        self.ordering_dot_prods = []
        self.Rank_max_dot_prods = []
        self.ordering_cosines = []
        self.Rank_max_cosines = []
        self.ordering_levin_scores = []
        self.Rank_min_costs = []
        self.indexes_rank_data = [0]

        self.memory = Memory ()
        self.memory_v2 = MemoryV2 (self._ordering_folder, self._puzzle_dims, "only_add_solutions_to_new_puzzles")

        if not parameters.checkpoint:
            # print ("not checkpointing program")
            self.iteration = 1
            self.total_expanded = 0
            self.total_generated = 0
            self.budget = self._initial_budget
            self.start = time.time ()
            self.current_solved_puzzles = set ()
            self.last_puzzle = list (self._states)[-1]  # self._states is a dictionary of puzzle_file_name, puzzle
            self.start_while = time.time ()

        else:
            print ("checkpointing program")
            self.iteration, self.total_expanded, self.total_generated, self.budget, self.current_solved_puzzles, \
            self.last_puzzle, self.start, self.start_while = restore_while_loop_state (self._puzzle_dims)
            # TODO: only need to restore before while loop starts.
            # TODO: because: pretend that program ends at end of while loop iteration -- then we need to just go through the inside of loop, as we would have without breakpoint

        self.while_loop_iter = 1
        self.sum_weights = tf.zeros((128, 4))

        compute_sum_weights(parameters.model_name, 1)

    def _call_solver(self, planner, nn_model):

        if self.parallelize_with_NN:
            chunk_size_heuristic = math.ceil (len (self._batch_problems) / (self._ncpus * 4))
            with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                args = ((state, name, self.budget, nn_model) for name, state in self._batch_problems.items ())
                results = executor.map (planner.search_for_learning, args, chunksize=chunk_size_heuristic)

            for result in results:
                has_found_solution = result[0]
                trajectory = result[1]  # the solution trajectory
                self.total_expanded += result[2]  # ??
                self.total_generated += result[3]  # ??
                puzzle_name = result[4]
                # # TODO: for debug:
                # if '2x2_' in name and not already_skipped:
                #     already_skipped = True
                #     print ("inside clause -- puzzle_name", puzzle_name)
                #     continue
                # end debug

                if has_found_solution:
                    self.memory.add_trajectory (
                        trajectory)  # stores trajectory object into a list (the list contains instances of the Trajectory class)

                if has_found_solution and puzzle_name not in self.current_solved_puzzles:
                    self._number_solved += 1
                    self.current_solved_puzzles.add (puzzle_name)
                    self.memory_v2.add_trajectory (trajectory, puzzle_name)
                    self._P += [puzzle_name]
                    self._n_P += 1
                    # memory_v2 -- only captures the new solutions for new puzzles
                    # this means that we would save the solutions found the first time they are solved (not the new solutions)
                    # P only contains the names of puzzles that are recently solved
                    # if a puzzle was solved before (under different weights, or a different budget), then we do not add the puzzle to P

            print ("")
            print ("time spent on for loop so far = ", time.time () - self._s_for_loop)
            print ("num puzzles we still need to try to solve =",
                   self._num_states_still_to_try_to_solve)
            print ("number of puzzles we have tried to solve already", (self._j * self._batch_size))
            print ("puzzles solved so far (in total, over all while loop/for loop iterations)",
                   len (self.current_solved_puzzles))
            print ("number of puzzles solved in the current while loop iteration =", self._number_solved)
            print ("")
        else:
            s_solve_puzzles = time.time ()
            for name, state in self._batch_problems.items ():
                args = (state, name, self.budget, nn_model)
                # with tf.device ('/GPU:0'):
                has_found_solution, trajectory, total_expanded, total_generated, puzzle_name = \
                    planner.search_for_learning (args)
                self.total_expanded += total_expanded
                self.total_generated += total_generated
                if has_found_solution:
                    self.memory.add_trajectory (trajectory)

                if has_found_solution and puzzle_name not in self.current_solved_puzzles:
                    self._number_solved += 1
                    self.current_solved_puzzles.add (puzzle_name)
                    self.memory_v2.add_trajectory (trajectory, puzzle_name)
                    self._P += [puzzle_name]
                    self._n_P += 1
            e_solve_puzzles = time.time ()
            print ("time to solve batch of ", self._batch_size, "= ", e_solve_puzzles - s_solve_puzzles)

    def compute_debug_data(self, nn_model):
        # we only compute the cosine data with puzzles that are newly solved (with current weights and current budget).
        # I think this is fine. Otherwise, if P += [puzzle_name] was in the first "if statement", then we would be
        # computing the cosines of all puzzles (whether they were solved previously or not)
        if self._n_P > 0:
            # if self.use_GPUs:
            # with tf.device ('/GPU:0'):
            batch_images_P, batch_actions_P = retrieve_batch_data_solved_puzzles (self._P, self.memory_v2)
            # function retrieve_batch_data_solved_puzzles also stores images_P and
            # actions_P in dictionary (for each puzzle)

            cosine_P, dot_prod_P, new_metric_P, theta_diff, grads_P_l2 = compute_cosines (nn_model, self._models_folder,
                                                                              None, False, batch_images_P,
                                                                              batch_actions_P, self.while_loop_iter)  # sum_weights
            # TODO: if you want to compare Final_theta to theta_i, 3rd argument must be None
            #         if you want to compare theta_{i+1} to theta_i, 3rd argument must be while_loop_iter
            levin_cost_P, average_levin_cost_P, training_loss_P = compute_levin_cost (batch_images_P, batch_actions_P,
                                                                                      nn_model)

            # TODO: uncomment the following:
            argmax_p_dot_prods, Rank_dot_prods, argmax_p_cosines, Rank_cosines, argmax_p_new_metric, Rank_new_metric \
                = compute_rank (self._P, nn_model, theta_diff, self.memory_v2, self._ncpus, 19, self._n_P,
                                self.parallelize_with_NN)

            argmin_p_levin_score, Rank_levin_scores = compute_rank_mins (self._P, nn_model, self.memory_v2,
                                                                         self._ncpus, 19, self._n_P,
                                                                         self.parallelize_with_NN)

            self.ordering_new_metric.append (argmax_p_new_metric)
            self.Rank_max_new_metric.append (Rank_new_metric)
            self.ordering_dot_prods.append (argmax_p_dot_prods)
            self.Rank_max_dot_prods.append (Rank_dot_prods)
            self.ordering_cosines.append (argmax_p_cosines)
            self.Rank_max_cosines.append (Rank_cosines)

            self.ordering_levin_scores.append (argmin_p_levin_score)
            self.Rank_min_costs.append (Rank_levin_scores)
            print ("len (ordering_dot_prods) =", len (self.ordering_dot_prods))

            self._new_metric_data_P.append (new_metric_P)
            self._cosine_data_P.append (cosine_P)
            self._dot_prod_data_P.append (dot_prod_P)
            self._levin_costs_P.append (levin_cost_P)
            self._average_levin_costs_P.append (average_levin_cost_P)
            self._training_losses_P.append (training_loss_P)
            self._grads_l2_P.append (grads_P_l2)

            print ("len(self._P) =", len (self._P))
            print ("while_loop_iter =", self.while_loop_iter)

            idx = self.indexes_rank_data[-1]
            self.indexes_rank_data.append (idx + self._n_P)
            if self.indexes_rank_data[0] == 0:
                self.indexes_rank_data = self.indexes_rank_data[1:]

    def _solve_uniform_online(self, planner, nn_model, parameters):
        print("here")
        self.initialize(parameters)

        while len (self.current_solved_puzzles) < self._number_problems:
            self._number_solved = 0
            self._batch_problems = {}
            self._s_for_loop = time.time ()
            # loop-invariant: on each loop iteration, we process self._batch_size puzzles that we solve
            # with current budget and train the NN on solved instances self._gradient_descent_steps times
            # (on the batch of solved puzzles)
            # before the for-loop starts, batch_problems = {}, number_solved = 0, at_least_one_got_solved = False
            # TODO: we would open the for-loop checkpoint file here

            self._j = 0
            self._P = []
            self._n_P = 0
            for name, state in self._states.items ():  # iterate through all the puzzles, try to solve however many you have with a current budget
                # at the start of each for loop, batch_problems is either empty, or it contains exactly self._batch_size puzzles
                # number_solved is either 0 or = the number of puzzles solved with the current budget
                # at_least_one_got_solved is either still False (no puzzle got solved with current budget) or is True
                # memory has either 0 solution trajectories or at least one solution trajectory
                self._batch_problems[name] = state

                if len (self._batch_problems) < self._batch_size and self.last_puzzle != name:
                    continue  # we only proceed if the number of elements in batch_problems == self._batch_size

                self._j += 1
                self._num_states_still_to_try_to_solve = self._number_problems - (self._j * self._batch_size)

                self._call_solver(planner, nn_model)
                print("self.total_expanded, self.total_generated", self.total_expanded, self.total_generated)

                self._batch_problems.clear ()  # at the end of the bigger for loop, batch_problems == {}

            print ("time for for loop to go over all puzzles =", time.time () - self._s_for_loop)
            # before training, we compute the debug data:
            self.compute_debug_data (nn_model)

            print ("")
            print ("NOW TRAINING -----------------")
            # if self.use_GPUs:
            # with tf.device ('/GPU:0'):
            if self.memory.number_trajectories () > 0:  # if you have solved at least one puzzle with given budget, then:
                                                   # before the for loop starts, memory has at least 1 solution trajectory
                                                   # and we have not yet trained the NN with  any of the puzzles solved
                                                   # with current budget and stored in memory
                for _ in range (self._gradient_steps):
                    loss = nn_model.train_with_memory (self.memory)
                    print ('Loss: ', loss)
                self.memory.clear ()
                self.while_loop_iter += 1
            print ("finished training -----------")
            print ("")
            print ("")
            # either we solved at least one puzzle with current budget, or 0 puzzles with current budget.
            # if we did solve at east one of the self._batch_size puzzles puzzles in the batch, then we train the NN
            # self._gradient_steps times with however many puzzles solved
            end = time.time ()

            self.store_txt_files_and_check_budget(end)

            end_while = time.time ()
            print ("time for while-loop iter =", end_while - self.start_while)
            print ("")

            if self._number_solved == 0:
                self.budget *= 2
                print ('Budget: ', self.budget)
                continue

        self.save_data(nn_model)

    def store_txt_files_and_check_budget(self, end):
        with open (join (self._log_folder, 'training_bootstrap_' + self._model_name), 'a') as results_file:
            results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (self.while_loop_iter,
                                                                                           self.iteration,
                                                                                           self._number_solved,
                                                                                           self._number_problems - len (
                                                                                               self.current_solved_puzzles),
                                                                                           self.budget,
                                                                                           self.total_expanded,
                                                                                           self.total_generated,
                                                                                           end - self.start)))
            results_file.write ('\n')

        print ('Number solved: ', self._number_solved, ' Number still to solve', self._number_problems -
               len (self.current_solved_puzzles))

        if self._number_solved > 0:
            unsolved_puzzles = self._all_puzzle_names.difference (self.current_solved_puzzles)
            with open (join (self._log_folder, 'unsolved_puzzles_' + self._model_name), 'a') as file:
                for puzzle in unsolved_puzzles:  # current_solved_puzzles:
                    file.write ("%s," % puzzle)
                file.write ('\n')
                file.write ('\n')

    def save_data(self, nn_model):
        # save_data_to_disk currently in write mode #TODO: should it be in append mode?
        print ("We are done if: len (current_solved_puzzles) == self._number_problems",
               len (self.current_solved_puzzles) == self._number_problems)
        if len (self.current_solved_puzzles) == self._number_problems:  # and iteration % 2 != 0.0:
            save_data_to_disk (self.Rank_max_new_metric,
                               join (self._ordering_folder,
                                     'Rank_NewMetric_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.Rank_max_dot_prods,
                               join (self._ordering_folder,
                                     'Rank_MaxDotProd_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.Rank_max_cosines,
                               join (self._ordering_folder,
                                     'Rank_MaxCosines_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.Rank_min_costs,
                               join (self._ordering_folder,
                                     'Rank_MinLevinCost_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.ordering_new_metric,
                               join (self._ordering_folder,
                                     'Ordering_NewMetric_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.ordering_dot_prods,
                               join (self._ordering_folder,
                                     'Ordering_DotProds_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.ordering_cosines,
                               join (self._ordering_folder,
                                     'Ordering_Cosines_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.ordering_levin_scores,
                               join (self._ordering_folder,
                                     'Ordering_LevinScores_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self.indexes_rank_data,
                               join (self._ordering_folder,
                                     'Idxs_rank_data_BFS_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._new_metric_data_P,
                               join (self._ordering_folder,
                                     'New_metric_over_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._grads_l2_P,
                               join (self._ordering_folder,
                                     'New_metric_over_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._cosine_data_P,
                               join (self._ordering_folder,
                                     'L2_Grad_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._dot_prod_data_P,
                               join (self._ordering_folder,
                                     'Dot_Prod_over_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._levin_costs_P,
                               join (self._ordering_folder,
                                     'Levin_Cost_over_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._average_levin_costs_P,
                               join (self._ordering_folder,
                                     'Average_Levin_Cost_over_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            save_data_to_disk (self._training_losses_P,
                               join (self._ordering_folder,
                                     'Training_Loss_over_P_theta_n-theta_i_' + str (self._puzzle_dims) + ".pkl"))

            nn_model.save_weights (join (self._models_folder, "Final_weights_n.h5"))


    def solve_problems(self, planner, nn_model, parameters):
        print ("in bootstrap solve_problems -- now going to call _solve_uniform_online")
        if self._scheduler == 'online':
            self._solve_uniform_online (planner, nn_model, parameters)
