import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from os.path import join
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math
import numpy as np
import tensorflow as tf

np.random.seed (1)
# random.seed (1)

from models.memory import Memory, MemoryV2
from compute_cosines import retrieve_batch_data_solved_puzzles, check_if_data_saved, retrieve_all_batch_images_actions,\
    compute_cosines, save_data_to_disk, find_argmax, compute_levin_cost, find_minimum, compute_rank, compute_rank_mins
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
    def __init__(self, states, output, scheduler, use_GPUs=False, ncpus=1, initial_budget=2000, gradient_steps=10):
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

        self._log_folder = 'logs_large/' + self._puzzle_dims  #+ "_debug_data"
        self._models_folder = 'trained_models_large/BreadthFS_' + self._model_name  #+ "_debug_data"
        self._ordering_folder = 'solved_puzzles/puzzles_' + self._puzzle_dims  #+ "_debug_data"

        if not os.path.exists (self._models_folder):
            os.makedirs (self._models_folder, exist_ok=True)

        if not os.path.exists (self._log_folder):
            os.makedirs (self._log_folder, exist_ok=True)

        if not os.path.exists (self._ordering_folder):
            os.makedirs (self._ordering_folder, exist_ok=True)

        # self._cosine_data = []
        # self._dot_prod_data = []
        # self._levin_costs = []
        # self._average_levin_costs = []
        # self._training_losses = []
        self.use_GPUs = use_GPUs

    def map_function(self, data):
        gbs = data[0]
        nn_model = data[1]
        max_steps = data[2]

        return gbs.solve (nn_model, max_steps)

    def _solve_uniform_online(self, planner, nn_model, parameters):
        print("inside _solve_uniform_online")
        parallelize_with_NN = bool(int(parameters.parallelize_with_NN))
        print ("parallelize with NN? ", parallelize_with_NN)
        print ("")

        # TODO:
        # step 1: check if we have already stored the solution trajectories and images of all the puzzles
        # already_saved_data = check_if_data_saved (self._puzzle_dims) # -- checks if we already created the file
        # 'Solved_Puzzles_Batch_Data' and saved in the folder: batch_folder = 'solved_puzzles/' + puzzle_dims + "/"

        ordering_dot_prods = []
        Rank_max_dot_prods = []
        ordering_cosines = []
        Rank_max_cosines = []

        ordering_levin_scores = []
        Rank_min_costs = []
        indexes_rank_data = [0]

        memory = Memory ()
        memory_v2 = MemoryV2 (self._ordering_folder, self._puzzle_dims, "only_add_solutions_to_new_puzzles")

        if not parameters.checkpoint:
            print("not checkpointing program")
            iteration = 1
            total_expanded = 0
            total_generated = 0
            budget = self._initial_budget
            start = time.time ()
            current_solved_puzzles = set ()
            last_puzzle = list (self._states)[-1]  # self._states is a dictionary of puzzle_file_name, puzzle
            start_while = time.time()

        else:
            print("checkpointing program")
            iteration, total_expanded, total_generated, budget, current_solved_puzzles, last_puzzle, start, start_while \
                = restore_while_loop_state(self._puzzle_dims) # TODO: only need to restore before while loop starts.
            # #TODO: because: pretend that program ends at end of while loop iteration -- then we need to just go through the inside of loop, as we would have without breakpoint

        print ("")
        # # TODO: debug
        # already_skipped = False
        # # end debug
        while len (current_solved_puzzles) < self._number_problems:
            number_solved = 0
            batch_problems = {}
            s_for_loop = time.time ()
            # loop-invariant: on each loop iteration, we process self._batch_size puzzles that we solve
            # with current budget and train the NN on solved instances self._gradient_descent_steps times
            # (on the batch of solved puzzles)
            # before the for-loop starts, batch_problems = {}, number_solved = 0, at_least_one_got_solved = False
            # TODO: we would open the for-loop checkpoint file here
            # already_restored = restore_for_loop_state ()

            j = 0
            P = []
            n_P = 0
            for name, state in self._states.items ():  # iterate through all the puzzles, try to solve however many you have with a current budget
                # at the start of each for loop, batch_problems is either empty, or it contains exactly self._batch_size puzzles
                # number_solved is either 0 or = the number of puzzles solved with the current budget
                # at_least_one_got_solved is either still False (no puzzle got solved with current budget) or is True
                # memory has either 0 solution trajectories or at least one solution trajectory
                batch_problems[name] = state

                if len (batch_problems) < self._batch_size and last_puzzle != name:
                    continue  # we only proceed if the number of elements in batch_problems == self._batch_size

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
                        total_expanded += result[2]  # ??
                        total_generated += result[3]  # ??
                        puzzle_name = result[4]
                        # print("puzzle name", puzzle_name)
                        # # TODO: for debug:
                        # if '2x2_' in name and not already_skipped:
                        #     already_skipped = True
                        #     continue
                        # # end debug

                        if has_found_solution:
                            memory.add_trajectory (trajectory)  # stores trajectory object into a list (the list contains instances of the Trajectory class)
                            # memory_v2.add_trajectory (trajectory, puzzle_name)

                        if has_found_solution and puzzle_name not in current_solved_puzzles:
                            number_solved += 1
                            current_solved_puzzles.add (puzzle_name)
                            memory_v2.add_trajectory (trajectory, puzzle_name)
                            P += [puzzle_name]
                            n_P += 1
                            # TODO: memory_v2 -- only capture the new solutions for new puzzles
                            # this means that we would save the solutions found the first time they are solved (not the new solutions)
                            # P only contains the names of puzzles that are recently solved
                            # if a puzzle was solved before (under different weights, or a different budget), then we do not add the puzzle to P

                    print("")
                    print ("time spent on for loop so far = ", time.time () - s_for_loop)
                    print ("num puzzles we still need to try to solve =",
                           num_states_still_to_try_to_solve)
                    print ("number of puzzles we have tried to solve already", (j * self._batch_size))
                    print ("puzzles solved so far (in total, over all while loop/for loop iterations)",
                           len (current_solved_puzzles))
                    print("number of puzzles solved in the current while loop iteration =", number_solved)
                    print("")
                else:
                    s_solve_puzzles = time.time()
                    for name, state in batch_problems.items ():
                        args = (state, name, budget, nn_model)
                        # with tf.device ('/GPU:0'):
                        has_found_solution, trajectory, total_expanded, total_generated, puzzle_name = planner.search_for_learning(args)
                        if has_found_solution:
                            memory.add_trajectory (trajectory)

                        if has_found_solution and puzzle_name not in current_solved_puzzles:
                            number_solved += 1
                            current_solved_puzzles.add (puzzle_name)
                            memory_v2.add_trajectory (trajectory, puzzle_name)
                            P += [puzzle_name]
                            n_P += 1
                    e_solve_puzzles = time.time()
                    print("time to solve batch of ", self._batch_size, "= ", e_solve_puzzles - s_solve_puzzles)

                batch_problems.clear ()  # at the end of the bigger for loop, batch_problems == {}

            print ("time for for loop to go over all puzzles =", time.time () - s_for_loop)

            # before training, we compute the debug data
            # we only compute the cosine data with puzzles that are newly solved (with current weights and current budget).
            # I think this is fine. Otherwise, if P += [puzzle_name] was in the first "if statement", then we would be
            # computing the cosines of all puzzles (whether they were solved previously or not)
            if n_P > 0:
                # if self.use_GPUs:
                # with tf.device ('/GPU:0'):
                retrieve_batch_data_solved_puzzles (P, memory_v2)  #TODO: used to return: batch_images_P, batch_actions_P
                # retrieve_batch_data_solved_puzzles also stores images_P and actions_P in dictionary (for each puzzle)

                theta_diff = compute_cosines (nn_model, self._models_folder)  # used to return: cosine, dot_prod, theta_diff
                # levin_cost, average_levin_cost, training_loss = compute_levin_cost (batch_images_P, batch_actions_P, nn_model)

                argmax_p_dot_prods, Rank_dot_prods, argmax_p_cosines, Rank_cosines = compute_rank (P, nn_model, theta_diff, memory_v2, self._ncpus, 19, n_P, parallelize_with_NN)

                argmin_p_levin_score, Rank_levin_scores = compute_rank_mins (P, nn_model, memory_v2, self._ncpus, 19, n_P, parallelize_with_NN)  # How is the ordering different if we use argmin? How is the ranking different?

                # self._cosine_data.append(cosine)
                # self._dot_prod_data.append(dot_prod)
                # self._levin_costs.append(levin_cost)
                # self._average_levin_costs.append(average_levin_cost)
                # self._training_losses.append(training_loss)

                ordering_dot_prods.append(argmax_p_dot_prods)
                Rank_max_dot_prods.append(Rank_dot_prods)
                ordering_cosines.append(argmax_p_cosines)
                Rank_max_cosines.append(Rank_cosines)

                Rank_min_costs.append (Rank_levin_scores)
                ordering_levin_scores.append (argmin_p_levin_score)
                print ("len(P) =", len (P))
                print("len (ordering_dot_prods) =", len(ordering_dot_prods))

                idx = indexes_rank_data[-1]
                indexes_rank_data.append(idx + n_P)
                if indexes_rank_data[0] == 0:
                    indexes_rank_data = indexes_rank_data[1:]
                print ("indexes_rank_data", indexes_rank_data)

            print("")
            print("")
            print("NOW TRAINING -----------------")
            # if self.use_GPUs:
            # with tf.device ('/GPU:0'):
            if memory.number_trajectories () > 0:  # if you have solved at least one puzzle with given budget, then:
                # before the for loop starts, memory has at least 1 solution trajectory and we have not yet trained the NN with
                # any of the puzzles solved with current budget and stored in memory
                epsilon = 0.1
                for _ in range (self._gradient_steps):
                    loss = nn_model.train_with_memory (memory)
                    print ('Loss: ', loss)
                    if loss < epsilon:
                        break
                memory.clear ()
                # nn_model.save_weights (join (self._models_folder, "i_th_weights.h5"))
            print("finished training -----------")
            print("")
            print("")
            # either we solved at least one puzzle with current budget, or 0 puzzles with current budget.
            # if we did solve at east one of the self._batch_size puzzles puzzles in the batch, then we train the NN
            # self._gradient_steps times with however many puzzles solved

            end = time.time ()
            with open (join (self._log_folder, 'training_bootstrap_' + self._model_name), 'a') as results_file:
                results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (iteration,
                                                                                         number_solved,
                                                                                         self._number_problems - len (
                                                                                             current_solved_puzzles),
                                                                                         budget,
                                                                                         total_expanded,
                                                                                         total_generated,
                                                                                         end - start)))
                results_file.write ('\n')

            print ('Number solved: ', number_solved, ' Number still to solve', self._number_problems -
                   len (current_solved_puzzles))
            # d[budget] += 1
            if number_solved == 0:
                budget *= 2
                print ('Budget: ', budget)
                # d[budget] = 1
                continue

            unsolved_puzzles = self._all_puzzle_names.difference (current_solved_puzzles)
            with open (join (self._log_folder, 'unsolved_puzzles_' + self._model_name), 'a') as file:
                for puzzle in unsolved_puzzles:  # current_solved_puzzles:
                    file.write ("%s," % puzzle)
                file.write ('\n')
                file.write ('\n')

            end_while = time.time()
            print("time for while-loop iter =", end_while - start_while)
            print("")

            # print("")
            # # TODO: add breakpoint here -- save --  in current while loop iteration: -- pretend that the following will happen right before breakpoint
            # # s_ckpt = time.time()
            # # if iteration % 1 == 0.0:  # iteration % i == 0 --> i-1 iterations already went through, because we start with i = 1
            # print("checkpoint current while loop data")
            # save_data_to_disk (Rank_max_dot_prods, join (self._ordering_folder, 'Rank_MaxDotProd_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # save_data_to_disk (Rank_max_cosines, join (self._ordering_folder, 'Rank_MaxCosines_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # save_data_to_disk (Rank_min_costs, join (self._ordering_folder, 'Rank_MinLevinCost_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # save_data_to_disk (indexes_rank_data, join (self._ordering_folder, 'Idxs_rank_data_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # save_data_to_disk (ordering_dot_prods, join (self._ordering_folder, 'Ordering_DotProds_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # save_data_to_disk (ordering_cosines, join (self._ordering_folder, 'Ordering_Cosines_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # save_data_to_disk (ordering_levin_scores, join (self._ordering_folder, 'Ordering_LevinScores_BFS_' + str (self._puzzle_dims) + ".pkl"))
            # nn_model.save_weights (join (self._models_folder, "checkpointed_weights.h5"))  # TODO: we need to do this in case memory.number_trajectories () == 0
            #
            # # if we are checkpointing, then every i iterations, we terminate the program:
            # save_while_loop_state (self._puzzle_dims, iteration, total_expanded, total_generated, budget,
            #                        current_solved_puzzles, last_puzzle, start, start_while)
            # if parameters.checkpoint:
            #     break
            # # print("time to save all checkpointed data =", time.time() - s_ckpt)
            #
            # Rank_max_dot_prods = []
            # Rank_max_cosines = []
            # Rank_min_costs = []
            # indexes_rank_data = [0]
            # ordering_dot_prods = []
            # ordering_cosines = []
            # ordering_levin_scores = []
            # print("")

                # TODO !!!! save_data_to_disk must use append mode, not write mode
        print("We are done if: len (current_solved_puzzles) == self._number_problems", len (current_solved_puzzles) == self._number_problems)
        if len (current_solved_puzzles) == self._number_problems: # and iteration % 2 != 0.0:
            save_data_to_disk (Rank_max_dot_prods, join (self._ordering_folder, 'Rank_MaxDotProd_BFS_' + str (self._puzzle_dims) + ".pkl"))
            save_data_to_disk (Rank_max_cosines, join (self._ordering_folder, 'Rank_MaxCosines_BFS_' + str (self._puzzle_dims) + ".pkl"))
            save_data_to_disk (Rank_min_costs, join (self._ordering_folder, 'Rank_MinLevinCost_BFS_' + str (self._puzzle_dims) + ".pkl"))
            save_data_to_disk (indexes_rank_data, join (self._ordering_folder, 'Idxs_rank_data_BFS_' + str (self._puzzle_dims) + ".pkl"))
            save_data_to_disk (ordering_dot_prods, join (self._ordering_folder, 'Ordering_DotProds_BFS_' + str (self._puzzle_dims) + ".pkl"))
            save_data_to_disk (ordering_cosines, join (self._ordering_folder, 'Ordering_Cosines_BFS_' + str (self._puzzle_dims) + ".pkl"))
            save_data_to_disk (ordering_levin_scores, join (self._ordering_folder, 'Ordering_LevinScores_BFS_' + str (self._puzzle_dims) + ".pkl"))
            nn_model.save_weights (join(self._models_folder, "Final_weights.h5"))  # nn_model.save_weights (join (self._models_folder, 'model_weights'))
            # memory_v2.save_data ()

    def solve_problems(self, planner, nn_model, parameters):
        print("in bootstrap solve_problems -- now going to call _solve_uniform_online")
        if self._scheduler == 'online':
            self._solve_uniform_online (planner, nn_model, parameters)