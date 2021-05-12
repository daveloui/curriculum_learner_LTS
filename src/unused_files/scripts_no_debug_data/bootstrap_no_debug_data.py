import os
import time
from os.path import join
from models.memory import Memory, MemoryV2
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math
import numpy as np

np.random.seed (1)
# random.seed (1)

from compute_cosines import retrieve_batch_data_solved_puzzles, check_if_data_saved, retrieve_all_batch_images_actions,\
    compute_cosines, save_data_to_disk, find_argmax, compute_levin_cost, find_minimum, compute_rank, compute_rank_mins
from save_while_for_loop_states import save_while_loop_state


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

    def _parallel_gbs(self, planner, nn_model):
        schedulers = []

        number_problems_per_cpu = math.ceil (self._number_problems / self._ncpus)

        states = {}
        counter_puzzles = 1
        for id_puzzle, instance in self._states.items ():
            if counter_puzzles > number_problems_per_cpu:
                gbs = GBS (states, planner)
                schedulers.append (gbs)
                counter_puzzles = 0
                states = {}

            states[id_puzzle] = instance
            counter_puzzles += 1

        if counter_puzzles > 0:
            gbs = GBS (states, planner)
            schedulers.append (gbs)

        #         print('Schedulers: ', schedulers)

        number_problems_solved = 0
        problems_solved_iteration = 0
        total_expanded = 0
        total_generated = 0

        start = time.time ()
        start_segment = start

        iteration = 1

        # while there are problems yet to be solved
        while number_problems_solved < self._number_problems:

            # invokes planning algorithm for solving the instance represented by node
            with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                args = ((gbs, nn_model, 10) for gbs in schedulers)
                results = executor.map (self.map_function, args)

            # collect the results of search for the states
            schedulers = []
            memory = Memory ()
            for result in results:
                memory.merge_trajectories (result[0])

                total_expanded += result[1]
                total_generated += result[2]

                problems_solved_iteration += result[3]
                number_problems_solved += result[3]

                gbs = result[4]
                schedulers.append (gbs)

            print ('Total number of problems solved: ', number_problems_solved)

            # if problems were solved in the previous batch, then use them to train the model
            if memory.number_trajectories () > 0:
                for _ in range (self._gradient_steps):
                    # perform a number of gradient descent steps
                    loss = nn_model.train_with_memory (memory)
                    print (loss)
                memory.clear ()
                # saving the weights the latest neural model
                nn_model.save_weights (join (self._models_folder, 'model_weights'))

            # it will report in the log file every 30 minutes of search
            end = time.time ()
            if end - start_segment > 1800:
                # readjusting elapsed time
                start_segment = end

                # logging details of the latest iteration
                with open (join (self._log_folder, 'training_bootstrap_' + self._model_name), 'a') as results_file:
                    results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (iteration,
                                                                                       problems_solved_iteration,
                                                                                       self._number_problems - number_problems_solved,
                                                                                       total_expanded,
                                                                                       total_generated,
                                                                                       end - start)))
                    results_file.write ('\n')

                iteration += 1
                problems_solved_iteration = 0

    def _solve_gbs(self, planner, nn_model):

        # counter for the number of iterations of the algorithm, which is marked by the number of times we train the model
        iteration = 1

        # number of problemsm solved in a given iteration of the procedure
        number_solved = 0

        # total number of nodes expanded
        total_expanded = 0

        # total number of nodes generated
        total_generated = 0

        # data structure used to store the solution trajectories
        memory = Memory ()

        # start timer for computing the running time of the iterations of the procedure
        start = time.time ()
        start_segment = start

        # open list of the scheduler
        open_list = []

        # dictionary storing the problem instances to be solved
        problems = {}
        has_solved_problem = [None]

        # populating the problems dictionary
        id_puzzle = 1
        for _, instance in self._states.items ():
            problems[id_puzzle] = instance
            id_puzzle += 1
            has_solved_problem.append (False)

            # create ProblemNode for the first puzzle in the list of puzzles to be solved
        node = ProblemNode (1, 1, problems[1])

        # insert such node in the open list
        heapq.heappush (open_list, node)

        # list containing all puzzles already solved
        closed_list = set ()

        # list of problems that will be solved in parallel
        problems_to_solve = {}

        # main loop of scheduler, iterate while there are problems still to be solved
        while len (open_list) > 0 and len (closed_list) < self._number_problems:

            # remove the first problem from the scheduler's open list
            node = heapq.heappop (open_list)

            # if the problem was already solved, then we bypass the solving part and
            # add the children of this node into the open list.
            if node.get_n () in closed_list:
                # if not halted
                if node.get_n () < self._number_problems:
                    # if not solved, then reinsert the same node with a larger budget into the open list
                    child = ProblemNode (node.get_k (),
                                         node.get_n () + 1,
                                         problems[node.get_n () + 1])
                    heapq.heappush (open_list, child)

                # if the first problem in the list then insert it with a larger budget
                if node.get_n () == 1:
                    # verifying whether there is a next puzzle in the list
                    if node.get_k () + 1 < self._kmax:
                        # create an instance of ProblemNode for the next puzzle in the list of puzzles.
                        child = ProblemNode (node.get_k () + 1,
                                             1,
                                             problems[1])
                        # add the node to the open list
                        heapq.heappush (open_list, child)
                continue

            # append current node in the list of problems to be solved
            problems_to_solve[node.get_n ()] = node

            # is there are at least k problems we will attempt to solve them in parallel
            if len (problems_to_solve) >= self._batch_size or len (open_list) == 0:
                # invokes planning algorithm for solving the instance represented by node
                with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                    args = ((p.get_instance (), p.get_n (), p.get_budget (), nn_model) for _, p in
                            problems_to_solve.items ())
                    results = executor.map (planner.search_for_learning, args)

                # collect the results of search for the states
                for result in results:
                    solved = result[0]
                    trajectory = result[1]
                    total_expanded += result[2]
                    total_generated += result[3]
                    puzzle_id = result[4]

                    # if not halted
                    if problems_to_solve[puzzle_id].get_n () < self._number_problems:
                        # if not solved, then reinsert the same node with a larger budget into the open list
                        child = ProblemNode (problems_to_solve[puzzle_id].get_k (),
                                             puzzle_id + 1,
                                             problems[puzzle_id + 1])
                        heapq.heappush (open_list, child)

                    if solved:
                        # if it has solved, then add the puzzle's name to the closed list
                        closed_list.add (puzzle_id)
                        # store the trajectory as training data
                        memory.add_trajectory (trajectory)
                        # mark problem as solved
                        has_solved_problem[puzzle_id] = True
                        # increment the counter of problems solved, for logging purposes
                        number_solved += 1

                    # if this is the puzzle's first trial, then share its computational budget with the next puzzle in the list
                    if puzzle_id == 1:
                        # verifying whether there is a next puzzle in the list
                        if problems_to_solve[puzzle_id].get_k () + 1 < self._kmax:
                            # create an instance of ProblemNode for the next puzzle in the list of puzzles.
                            child = ProblemNode (problems_to_solve[puzzle_id].get_k () + 1,
                                                 1,
                                                 problems[1])
                            # add the node to the open list
                            heapq.heappush (open_list, child)

                # clear the problems to solve
                problems_to_solve.clear ()

                if memory.number_trajectories () > 0:
                    # perform a number of gradient descent steps
                    for _ in range (self._gradient_steps):
                        loss = nn_model.train_with_memory (memory)
                        print (loss)

                    # remove current trajectories from memory
                    memory.clear ()

                    # saving the weights the latest neural model
                    nn_model.save_weights (join (self._models_folder, 'model_weights'))

            # if the number of attempts for solving problems is equal to the number of remaining problems and
            # if the procedure solved problems a new problem, then perform learning
            if time.time () - start_segment > 1800 and number_solved > 0:
                # time required in this iteration of the algorithm
                end = time.time ()

                # readjusting elapsed time
                start_segment = end

                # logging details of the latest iteration
                with open (join (self._log_folder, 'training_bootstrap_' + self._model_name), 'a') as results_file:
                    results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (iteration,
                                                                                       number_solved,
                                                                                       self._number_problems - len (
                                                                                           closed_list),
                                                                                       total_expanded,
                                                                                       total_generated,
                                                                                       end - start)))
                    results_file.write ('\n')

                # set the number of problems solved and trials to zero and increment the iteration counter
                number_solved = 0
                iteration += 1

                # if the system solves all instances and there are new instances to learn from, then log details and train the model
        if number_solved > 0:
            # time required in this iteration of the algorithm
            end = time.time ()
            # logging details of the latest iteration
            with open (join (self._log_folder, 'training_bootstrap_' + self._model_name), 'a') as results_file:
                results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (iteration,
                                                                                   number_solved,
                                                                                   self._number_problems - len (
                                                                                       closed_list),
                                                                                   total_expanded,
                                                                                   total_generated,
                                                                                   end - start)))
                results_file.write ('\n')

    def _solve_uniform_online(self, planner, nn_model, parameters):
        print("inside _solve_uniform_online")
        use_epsilon = bool(int(parameters.use_epsilon))
        print("use_epsilon =", use_epsilon, " type(use_epsilon) =", type(use_epsilon))
        print("")

        # tODO: assumption: the solution of each puzzle is unique -- therefore, each time we train the NN and then solve the puzzles again,
        # we will not need to recompute the solution trajectories after each trainign of theta

        # step 1: check if we have already stored the solution trajectories and images of all the puzzles
        # already_saved_data = check_if_data_saved (self._puzzle_dims)
        # print ("already_saved_data", already_saved_data)

        if not parameters.checkpoint:
            print("not checkpointing program")
            iteration = 1
            total_expanded = 0
            total_generated = 0

            budget = self._initial_budget
            memory = Memory ()
            memory_v2 = MemoryV2 (self._ordering_folder, self._puzzle_dims)  # added by FD
            start = time.time ()

            current_solved_puzzles = set ()
            last_puzzle = list (self._states)[-1]  # self._states is a dictionary of puzzle_file_name, puzzle

            # TODO: only save a ordering if you solved all puzzles
            ordering = []
            # if you did not solve any puzzles with current budget, then double the budget
            # d = {}
            # d[budget] = 1
            start_while = time.time()
            n_while = 0
            Rank_max_dot_prods = []
            Rank_min_costs = []

        else:
            print("checkpointing program")
            # restore_while_loop_state() # TODO: only need to restore before while loop starts.
            # #TODO: because: pretend that program ends at end of while loop iteration -- then we need to just go through the inside of loop, as we would have without breakpoint

        t_cos = -1
        while len (current_solved_puzzles) < self._number_problems:

            n_while += 1
            at_least_one_got_solved = False
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

                with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                    args = ((state, name, budget, nn_model) for name, state in batch_problems.items ())
                    results = executor.map (planner.search_for_learning, args)
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
                        at_least_one_got_solved = True

                        memory_v2.add_trajectory (trajectory, puzzle_name)

            # FD: once you have added everything to memory, train:
            if memory.number_trajectories () > 0:  # if you have solved at least one puzzle with given budget, then:
                # before the for loop starts, memory has at least 1 solution trajectory and we have not yet trained the NN with
                # any of the puzzles solved with current budget and stored in memory
                # if use_epsilon:
                loss = 1000000
                epsilon = 0.1
                while loss > epsilon:
                    loss = nn_model.train_with_memory (memory)
                    print('Loss: ', loss)
                # else:
                #     for _ in range (self._gradient_steps):
                #         loss = nn_model.train_with_memory (memory)
                #         print ('Loss: ', loss)

                memory.clear ()  # clear memory
                # nn_model.save_weights (join (self._models_folder, "i_th_weights.h5"))  # nn_model.save_weights (join (self._models_folder, 'model_weights'))

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

            iteration += 1

            unsolved_puzzles = self._all_puzzle_names.difference (current_solved_puzzles)
            with open (join (self._log_folder, 'unsolved_puzzles_' + self._model_name), 'a') as file:
                for puzzle in unsolved_puzzles:  # current_solved_puzzles:
                    file.write ("%s," % puzzle)
                file.write ('\n')
                file.write ('\n')

            # save the data we have so far
            if at_least_one_got_solved:
                memory_v2.save_data ()

            end_while = time.time()
            print("time for while-loop iter =", end_while - start_while)
            print("")

            # TODO: add breakpoint here -- save --  in current while loop iteration:
            if parameters.checkpoint and n_while % 50 == 0.0:
                # save_data_to_disk (self._dict_cos, join (self._log_folder, "dict_times_puzzles_for_cosine_data_" + self._model_name))
                nn_model.save_weights (join (self._models_folder,
                                             "checkpointed_weights.h5"))  # TODO: we need to do this in case memory.number_trajectories () < 0
                # save the data we have so far
                if at_least_one_got_solved: #TODO: not needed. If we solved at least one in current iteration, the memory_v2 got already saved
                    # TODO: if we solved none in current iteration, then memory_v2 got saved in previous iteration. Nothing new to save in current iter anyway
                    memory_v2.save_data ()  # not need this -- you can reinitialize the memory_v2 to be empty, when load checkpoint
                    # because we only need puzzles solved in current while-loop iteration, not all puzzles solved in all iterations!
            #     # TODO !!!! save_data_to_disk must use append mode, not write mode

        nn_model.save_weights (join(self._models_folder, "Final_weights_n-i.h5"))  # nn_model.save_weights (join (self._models_folder, 'model_weights'))
        memory_v2.save_data ()  # FD

    def _solve_uniform(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0

        budget = self._initial_budget
        memory = Memory ()
        start = time.time ()

        current_solved_puzzles = set ()

        while len (current_solved_puzzles) < self._number_problems:
            number_solved = 0

            with ProcessPoolExecutor (max_workers=self._ncpus) as executor:
                args = ((state, name, budget, nn_model) for name, state in self._states.items ())
                results = executor.map (planner.search_for_learning, args)
            for result in results:
                has_found_solution = result[0]
                trajectory = result[1]
                total_expanded += result[2]
                total_generated += result[3]
                puzzle_name = result[4]

                if has_found_solution:
                    memory.add_trajectory (trajectory)

                if has_found_solution and puzzle_name not in current_solved_puzzles:
                    number_solved += 1
                    current_solved_puzzles.add (puzzle_name)

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

            print ('Number solved: ', number_solved)
            if number_solved > 0:
                for _ in range (self._gradient_steps):
                    loss = nn_model.train_with_memory (memory)
                    print (loss)
                memory.clear ()

                nn_model.save_weights (join (self._models_folder, 'model_weights'))
            else:
                budget *= 2
                print ('Budget: ', budget)
                continue

            iteration += 1

    def solve_problems(self, planner, nn_model, parameters):
        if self._scheduler == 'gbs':
            self._solve_gbs (planner, nn_model)
        elif self._scheduler == 'online':
            self._solve_uniform_online (planner, nn_model, parameters)
        elif self._scheduler == 'pgbs':
            self._parallel_gbs (planner, nn_model)
        else:
            self._solve_uniform (planner, nn_model)