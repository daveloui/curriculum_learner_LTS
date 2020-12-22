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


class GBS:
    def __init__(self, states, planner):
        self._states = states
        self._number_problems = len (states)

        self._planner = planner

        # maximum budget
        self._kmax = 10

        # counter for the number of iterations of the algorithm, which is marked by the number of times we train the model
        self._iteration = 1

        # number of problemsm solved in a given iteration of the procedure
        self._number_solved = 0

        # total number of nodes expanded
        self._total_expanded = 0

        # total number of nodes generated
        self._total_generated = 0

        # data structure used to store the solution trajectories
        self._memory = Memory ()

        # open list of the scheduler
        self._open_list = []

        # dictionary storing the problem instances to be solved
        self._problems = {}

        self._last_tried_instance = [0 for _ in range (0, self._kmax + 1)]
        self._has_solved = {}  # [False for _ in range(0, self._number_problems + 1)]

        # populating the problems dictionary
        id_puzzle = 1
        for name, instance in self._states.items ():
            self._problems[id_puzzle] = (name, instance)
            self._has_solved[id_puzzle] = False
            id_puzzle += 1

            # create ProblemNode for the first puzzle in the list of puzzles to be solved
        node = ProblemNode (1, 1, self._problems[1][0], self._problems[1][1])

        # insert such node in the open list
        heapq.heappush (self._open_list, node)

        # list containing all puzzles already solved
        self._closed_list = set ()

    def run_prog(self, k, budget, nn_model):

        last_idx = self._last_tried_instance[k]
        idx = last_idx + 1

        while idx < self._number_problems + 1:
            if not self._has_solved[idx]:
                break
            idx += 1

        if idx > self._number_problems:
            return True, None, None, None, None

        self._last_tried_instance[k] = idx

        data = (self._problems[idx][1], self._problems[idx][0], budget, nn_model)
        is_solved, trajectory, expanded, generated, _ = self._planner.search_for_learning (data)

        if is_solved:
            self._has_solved[idx] = True

        return idx == self._number_problems, is_solved, trajectory, expanded, generated

    def solve(self, nn_model, max_steps):
        # counter for the number of steps in this schedule
        number_steps = 0

        # number of problems solved in this iteration
        number_solved_iteration = 0

        # reset the current memory
        self._memory.clear ()

        # main loop of scheduler, iterate while there are problems still to be solved
        while len (self._open_list) > 0 and len (self._closed_list) < self._number_problems:

            # remove the first problem from the scheduler's open list
            node = heapq.heappop (self._open_list)

            # if the problem was already solved, then we bypass the solving part and
            # add the children of this node into the open list.
            if False and node.get_n () in self._closed_list:
                # if not halted
                if node.get_n () < self._number_problems:
                    # if not solved, then reinsert the same node with a larger budget into the open list
                    child = ProblemNode (node.get_k (),
                                         node.get_n () + 1,
                                         self._problems[node.get_n () + 1][0],
                                         self._problems[node.get_n () + 1][1])
                    heapq.heappush (self._open_list, child)

                # if the first problem in the list then insert it with a larger budget
                if node.get_n () == 1:
                    # verifying whether there is a next puzzle in the list
                    if node.get_k () + 1 < self._kmax:
                        # create an instance of ProblemNode for the next puzzle in the list of puzzles.
                        child = ProblemNode (node.get_k () + 1,
                                             1,
                                             self._problems[1][0],
                                             self._problems[1][1])
                        # add the node to the open list
                        heapq.heappush (self._open_list, child)
                continue

            #             data = (node.get_instance(), node.get_name(), node.get_budget(), nn_model)
            #             solved, trajectory, expanded, generated, _ = self._planner.search_for_learning(data)
            has_halted, solved, trajectory, expanded, generated = self.run_prog (node.get_k (), node.get_budget (),
                                                                                 nn_model)

            # if not halted
            # if node.get_n() < self._number_problems:
            if not has_halted:
                self._total_expanded += expanded
                self._total_generated += generated

                # if not solved, then reinsert the same node with a larger budget into the open list
                child = ProblemNode (node.get_k (),
                                     node.get_n () + 1,
                                     self._problems[node.get_n () + 1][0],
                                     self._problems[node.get_n () + 1][1])
                heapq.heappush (self._open_list, child)

            if solved is not None and solved:
                # if it has solved, then add the puzzle's name to the closed list
                self._closed_list.add (node.get_n ())
                # store the trajectory as training data
                self._memory.add_trajectory (trajectory)
                # increment the counter of problems solved, for logging purposes
                self._number_solved += 1
                number_solved_iteration += 1

            # if this is the puzzle's first trial, then share its computational budget with the next puzzle in the list
            if node.get_n () == 1:
                # verifying whether there is a next puzzle in the list
                if node.get_k () + 1 < self._kmax:
                    # create an instance of ProblemNode for the next puzzle in the list of puzzles.
                    child = ProblemNode (node.get_k () + 1,
                                         1,
                                         self._problems[1][0],
                                         self._problems[1][1])
                    # add the node to the open list
                    heapq.heappush (self._open_list, child)

            # increment the number of problems attempted solve
            number_steps += 1

            # if exceeds the maximum of steps allowed, then return training data, expansions, generations
            if number_steps >= max_steps:
                return self._memory, self._total_expanded, self._total_generated, number_solved_iteration, self

        return self._memory, self._total_expanded, self._total_generated, number_solved_iteration, self


class Bootstrap:
    def __init__(self, states, output, scheduler, use_epsilon=False, ncpus=1, initial_budget=2000, gradient_steps=10):
                 # parallelize_with_NN=True):

        self._states = states
        self._model_name = output
        self._number_problems = len (states)

        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps
        #         self._k = ncpus * 3
        self._batch_size = 32
        self._kmax = 10
        self._scheduler = scheduler
        # self._parallelize_with_NN = parallelize_with_NN

        self._all_puzzle_names = set (states.keys ())  ## FD what do we use this for?
        self._puzzle_dims = self._model_name.split ('-')[0]  # self._model_name has the form
        # '<puzzle dimension>-<problem domain>-<loss name>'

        self._log_folder = 'logs_large/' + self._puzzle_dims + "_use_epsilon=" + str(use_epsilon)
        self._models_folder = 'trained_models_large/BreadthFS_' + self._model_name + "_" + "use_epsilon=" + str(use_epsilon)
        self._ordering_folder = 'solved_puzzles/puzzles_' + self._puzzle_dims + "_use_epsilon=" + str (use_epsilon)

        if not os.path.exists (self._models_folder):
            os.makedirs (self._models_folder, exist_ok=True)

        if not os.path.exists (self._log_folder):
            os.makedirs (self._log_folder, exist_ok=True)

        if not os.path.exists (self._ordering_folder):
            os.makedirs (self._ordering_folder, exist_ok=True)

        self._cosine_data = []
        self._dot_prod_data = []
        # self._dict_cos = {}
        self._levin_costs = []
        self._average_levin_costs = []
        self._training_losses = []

    def map_function(self, data):
        gbs = data[0]
        nn_model = data[1]
        max_steps = data[2]

        return gbs.solve (nn_model, max_steps)

    def _solve_uniform_online(self, planner, nn_model, parameters):
        print("inside _solve_uniform_online")
        print("parallelize with NN? ", bool(int(parameters.parallelize_with_NN)))
        use_epsilon = bool(int(parameters.use_epsilon))
        print("use_epsilon =", use_epsilon)
        print("")

        # TODO:
        # step 1: check if we have already stored the solution trajectories and images of all the puzzles
        # already_saved_data = check_if_data_saved (self._puzzle_dims) # -- checks if we already created the file
        # 'Solved_Puzzles_Batch_Data' and saved in the folder: batch_folder = 'solved_puzzles/' + puzzle_dims + "/"

        Rank_max_dot_prods = []
        Rank_min_costs = []
        ordering = [] # TODO: only save a ordering if you solved all puzzles
        memory = Memory ()
        memory_v2 = MemoryV2 (self._ordering_folder, self._puzzle_dims, "add_solutions_to_new_puzzles")  # added by FD

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

        t_cos = -1
        print ("len(current_solved_puzzles) =", len (current_solved_puzzles))
        print ("")
        while len (current_solved_puzzles) < self._number_problems:
            number_solved = 0
            batch_problems = {}
            for_loop_index = 0

            # loop-invariant: on each loop iteration, we process self._batch_size puzzles that we solve
            # with current budget and train the NN on solved instances self._gradient_descent_steps times
            # (on the batch of solved puzzles)
            # before the for-loop starts, batch_problems = {}, number_solved = 0, at_least_one_got_solved = False

            # TODO: we would open the for-loop checkpoint file here
            # already_restored = restore_for_loop_state ()

            for name, state in self._states.items ():  # iterate through all the puzzles, try to solve however many you have with a current budget
                for_loop_index += 1
                # at the start of each for loop, batch_problems is either empty, or it contains exactly self._batch_size puzzles
                # number_solved is either 0 or = the number of puzzles solved with the current budget
                # at_least_one_got_solved is either still False (no puzzle got solved with current budget) or is True
                # memory has either 0 solution trajectories or at least one solution trajectory
                batch_problems[name] = state

                if len (batch_problems) < self._batch_size and last_puzzle != name:
                    continue  # we only proceed if the number of elements in batch_problems == self._batch_size

                # once we have self._batch_size puzzles in batch_problems, we look for their solutions and train NN
                P = []
                n_P = 0
                if bool(int(parameters.parallelize_with_NN)):
                    with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                        args = ((state, name, budget, nn_model) for name, state in batch_problems.items ())
                        results = executor.map (planner.search_for_learning, args)
                    print("passed parallelization to search_for_learning")
                    for result in results:
                        has_found_solution = result[0]
                        trajectory = result[1]  # the solution trajectory
                        total_expanded += result[2]  # ??
                        total_generated += result[3]  # ??
                        puzzle_name = result[4]

                        if has_found_solution:
                            memory.add_trajectory (trajectory)  # stores trajectory object into a list (the list contains instances of the Trajectory class)
                            # memory_v2.add_trajectory (trajectory, puzzle_name)

                        if has_found_solution and puzzle_name not in current_solved_puzzles:
                            number_solved += 1
                            current_solved_puzzles.add (puzzle_name)
                            memory_v2.add_trajectory (trajectory, puzzle_name)  # TODO: we only capture the new solutions for new puzzles
                            #TODO: this means that we would save the solutions found the first time they are solved (not the new solutions)
                            P += [puzzle_name]  # only contains the names of puzzles that are recently solved
                            # if a puzzle was solved before (under different weights, or a different budget), then we do not
                            # add the puzzle to P
                            n_P += 1
                else:
                    for name, state in batch_problems.items ():
                        args = (state, name, budget, nn_model)
                        has_found_solution, trajectory, total_expanded, total_generated, puzzle_name = planner.search_for_learning(args)
                        if has_found_solution:
                            memory.add_trajectory (trajectory)

                        if has_found_solution and puzzle_name not in current_solved_puzzles:
                            number_solved += 1
                            current_solved_puzzles.add (puzzle_name)
                            memory_v2.add_trajectory (trajectory, puzzle_name)
                            P += [puzzle_name]
                            n_P += 1

            print("current_solved_puzzles", current_solved_puzzles)
            # assert n_P == number_solved
            # before training, we compute the cosines data
            if n_P > 0:
                # we only compute the cosine data with puzzles that are newly solved (with current weights and current budget).
                # I think this is fine. Otherwise, if P += [puzzle_name] was in the first "if statement", then we would be
                # computing the cosines of all puzzles (whether they were solved previously or not)
                t_cos += 1
                batch_images_P, batch_actions_P = retrieve_batch_data_solved_puzzles (P, memory_v2)  # also stores images_P and actions_P in dictionary (for each puzzle)

                # TODO: should we get rid of states, actions saved? I think so! Because after training, we might have new actions/states
                cosine, dot_prod, theta_diff = compute_cosines (batch_images_P, batch_actions_P, nn_model,
                                                                self._models_folder, parameters)
                self._cosine_data += [cosine]
                self._dot_prod_data += [dot_prod]
                levin_cost, average_levin_cost, training_loss = compute_levin_cost (batch_images_P, batch_actions_P,
                                                                                    nn_model)
                self._levin_costs += [levin_cost]
                self._average_levin_costs += [average_levin_cost]
                self._training_losses += [training_loss]

                argmax_p, Rank_sublist = compute_rank (P, nn_model, theta_diff, memory_v2, self._ncpus, 19, n_P)
                ordering += [argmax_p]
                Rank_max_dot_prods.append(Rank_sublist)
                argmin_p, Rank_sublist = compute_rank_mins (P, nn_model, memory_v2, self._ncpus, 19, n_P)  # How is the ordering different if we use argmin? How is the ranking different?
                Rank_min_costs.append (Rank_sublist)

                # argmax_p = find_argmax(P, nn_model, theta_diff, memory_v2, self._ncpus, 19, n_P)
                # print("argmax", argmax_p)
                # print("Rank_sublist", Rank_sublist)
                # print("ordering =", ordering)
                # print("Rank_max_dot_prods =", Rank_max_dot_prods)
                # print ("argmin_p", argmin_p)
                # print ("Rank_sublist", Rank_sublist)
                # print ("Rank_min_costs =", Rank_min_costs)

            # FD: once you have added everything to memory, train:
            if memory.number_trajectories () > 0:  # if you have solved at least one puzzle with given budget, then:
                # before the for loop starts, memory has at least 1 solution trajectory and we have not yet trained the NN with
                # any of the puzzles solved with current budget and stored in memory
                if use_epsilon:
                    loss = 1000000
                    epsilon = 0.1
                    while loss > epsilon:
                        loss = nn_model.train_with_memory (memory)
                        print('Loss: ', loss)
                else:
                    for _ in range (self._gradient_steps):
                        loss = nn_model.train_with_memory (memory)
                        print ('Loss: ', loss)

                memory.clear ()  # clear memory
                nn_model.save_weights (join (self._models_folder, "i_th_weights.h5"))  # nn_model.save_weights (join (self._models_folder, 'model_weights'))

            batch_problems.clear ()  # at the end of the bigger for loop, batch_problems == {}
            # either we solved at least one puzzle with current budget, or 0 puzzles with current budget.
            # if we did solve at east one of the self._batch_size puzzles puzzles in the batch, then we train the NN
            # self._gradient_steps times with however many puzzles solved
            print ("")


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

            iteration += 1

            # TODO: add breakpoint here -- save --  in current while loop iteration: -- pretend that the following will happen right before breakpoint
            if iteration % 3 == 0.0:  # 51 --> 50  iterations already happened
                save_data_to_disk (self._cosine_data, join (self._log_folder, "cosine_data_" + self._model_name))
                save_data_to_disk (self._dot_prod_data, join (self._log_folder, "dot_prod_data_" + self._model_name))
                save_data_to_disk (self._levin_costs, join (self._log_folder, "levin_cost_data_" + self._model_name))
                save_data_to_disk (self._average_levin_costs, join (self._log_folder, "aver_levin_cost_data_" + self._model_name))
                save_data_to_disk (self._training_losses, join(self._log_folder, "training_loss_data_" + self._model_name))
                save_data_to_disk (Rank_max_dot_prods, join (self._ordering_folder, 'Rank_MaxDotProd_BFS_' + str (self._puzzle_dims)))
                save_data_to_disk (Rank_min_costs, join (self._ordering_folder, 'Rank_MinLevinCost_BFS_' + str (self._puzzle_dims)))
                # TODO: get rid of the following (?):
                save_data_to_disk (ordering, join (self._ordering_folder, 'Ordering_BFS_' + str (self._puzzle_dims)))

                # save_data_to_disk (self._dict_cos, join (self._log_folder, "dict_times_puzzles_for_cosine_data_" + self._model_name))
                nn_model.save_weights (join (self._models_folder,
                                             "checkpointed_weights.h5"))  # TODO: we need to do this in case memory.number_trajectories () < 0

                memory_v2.save_data ()
                save_while_loop_state (self._puzzle_dims, iteration, total_expanded, total_generated, budget,
                                       current_solved_puzzles, last_puzzle, start, start_while)

                break

                # TODO !!!! save_data_to_disk must use append mode, not write mode
        print("len (current_solved_puzzles) == self._number_problems", len (current_solved_puzzles) == self._number_problems)
        if len (current_solved_puzzles) == self._number_problems and iteration % 3 != 0.0:
            save_data_to_disk (self._cosine_data, join (self._log_folder, "cosine_data_" + self._model_name))
            save_data_to_disk (self._dot_prod_data, join (self._log_folder, "dot_prod_data_" + self._model_name))
            save_data_to_disk (self._levin_costs, join (self._log_folder, "levin_cost_data_" + self._model_name))
            save_data_to_disk (self._average_levin_costs, join (self._log_folder, "aver_levin_cost_data_" + self._model_name))
            save_data_to_disk (self._training_losses, join (self._log_folder, "training_loss_data_" + self._model_name))
            save_data_to_disk (ordering, join (self._ordering_folder, 'Ordering_BFS_' + str(self._puzzle_dims)))
            save_data_to_disk (Rank_max_dot_prods, join (self._ordering_folder, 'Rank_MaxDotProd_BFS_' + str(self._puzzle_dims)))
            save_data_to_disk (Rank_min_costs, join (self._ordering_folder, 'Rank_MinLevinCost_BFS_' + str(self._puzzle_dims)))
            # save_data_to_disk (self._dict_cos, join (self._log_folder, "dict_times_puzzles_for_cosine_data_" + self._model_name))

            nn_model.save_weights (join(self._models_folder, "Final_weights.h5"))  # nn_model.save_weights (join (self._models_folder, 'model_weights'))
            memory_v2.save_data ()

    def solve_problems(self, planner, nn_model, parameters):
        print("in bootstrap solve_problems -- now going to call _solve_uniform_online")
        if self._scheduler == 'online':
            self._solve_uniform_online (planner, nn_model, parameters)