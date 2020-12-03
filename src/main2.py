import sys
import os
import time
from os import listdir
from os.path import isfile, join
from domains.witness import WitnessState
from search.bfs_levin import BFSLevin
from models.model_wrapper import KerasManager, KerasModel
from concurrent.futures.process import ProcessPoolExecutor
import argparse
from search.a_star import AStar
from search.gbfs import GBFS
from search.bfs_levin_mult import BFSLevinMult
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from search.puct import PUCT
from bootstrap import Bootstrap
from multiprocessing import set_start_method
import pickle

from game_state import GameState
from bootstrap_dfs_learning_planner import BootstrapDFSLearningPlanner

from compute_cosines import store_or_retrieve_batch_data_solved_puzzles, check_if_data_saved, \
    retrieve_all_batch_images_actions, compute_and_save_cosines

os.environ['PYTHONHASHSEED'] = str(1)


def search(states, planner, nn_model, ncpus, time_limit_seconds, search_budget=-1):
    """
    This function runs (best-first) Levin tree search with a learned policy on a set of problems
    """
    total_expanded = 0
    total_generated = 0
    total_cost = 0

    slack_time = 600

    solutions = {}

    for name, state in states.items ():
        state.reset ()
        solutions[name] = (-1, -1, -1, -1)

    start_time = time.time ()

    while len (states) > 0:
        with ProcessPoolExecutor (max_workers=ncpus) as executor:
            args = ((state, name, nn_model, search_budget, start_time, time_limit_seconds, slack_time) for name, state
                    in states.items ())
            results = executor.map (planner.search, args)
        for result in results:
            solution_depth = result[0]
            expanded = result[1]
            generated = result[2]
            running_time = result[3]
            puzzle_name = result[4]

            if solution_depth > 0:
                solutions[puzzle_name] = (solution_depth, expanded, generated, running_time)
                del states[puzzle_name]

            if solution_depth > 0:
                total_expanded += expanded
                total_generated += generated
                total_cost += solution_depth

        partial_time = time.time ()

        if partial_time - start_time + slack_time > time_limit_seconds or len (states) == 0 or search_budget >= 1000000:
            for name, data in solutions.items ():
                print ("{:s}, {:d}, {:d}, {:d}, {:.2f}".format (name, data[0], data[1], data[2], data[3]))
            return

        search_budget *= 2


def test_bootstrap_dfs_lvn_learning_planner(states, output, epsilon, beta, dropout, batch, nn_model):
    ordering_folder = "solved_puzzles/" + output.split("/")[0].split("_output")[0]
    if not os.path.exists (ordering_folder):
        os.makedirs (ordering_folder)
    filename_save_ordering = 'Ordering_DFS'
    filename_save_ordering = os.path.join (ordering_folder, filename_save_ordering + '.pkl')

    gradient_descent_steps = 1
    planner = BootstrapDFSLearningPlanner (nn_model, beta, dropout,
                                           batch)  # FD: creates an instance of the class "BootstrapDFSLearningPlanner"
    # models_folder = output + '_models'
    models_folder = 'trained_models_large/' + str('DepthFS_NN_weights_' + output)
    print(models_folder)
    if not os.path.exists (models_folder):
        os.makedirs (models_folder)

    state_budget = {}  # FD: empty dictionary containing the budget
    for file, state in states.items ():  # FD: recall: states is a dictionary containing the -- I suppose -- initial states of each puzzle
        state_budget[state] = 1  # FD: initialize budget to 1 for each state in the states dictionary

    unsolved_puzzles = states  # FD: initially all puzzles -- represented by their initial states in the "states" dictionary are unsolved
    id_solved = 1  # FD: ID of puzzles solved
    id_batch = 1

    with open (join (output + '_puzzle_names_ordering_bootstrap'), 'a') as results_file:
        results_file.write (("{:d}".format (id_batch)))
        results_file.write ('\n')

    start = time.time ()
    while len (unsolved_puzzles) > 0:  # FD: while there is at least one unsolved puzzle
        print("len (unsolved_puzzles)", len (unsolved_puzzles))
        number_solved = 0
        number_unsolved = 0
        current_unsolved_puzzles = {}

        solved_puzzles = []
        for file, state in unsolved_puzzles.items ():  # FD: take each file and initial puzzle_state
            state.clear_path ()
            has_found_solution, new_bound = planner.lvn_search_budget_for_learning (state, state_budget[state])
            if has_found_solution:
                cost = planner.get_solution_depth ()
                number_solved += 1
                id_solved += 1
                solved_puzzles += [file]

                with open (join (output + '_puzzle_names_ordering_bootstrap'), 'a') as results_file:
                    results_file.write (file + ', ' + str (
                        cost) + ' \n')  # FD:writes the line: "<file name>, <cost>" and moves to next line
            else:  # FD: planner has not yet found the solution
                if new_bound != -1:  # the planner has returned a new "budget" (new_bound) to solve the current puzzle?
                    state_budget[state] = new_bound
                else:
                    state_budget[state] = state_budget[state] + 1  # FD:increase budget by one
                number_unsolved += 1
                current_unsolved_puzzles[file] = state  # FD: create the k, v pair: <file_name>, <puzzle initial state>

        end = time.time ()  # FD: time that it takes to solve all the puzzles with current budget

        with open (join (output + '_log_training_bootstrap'), 'a') as results_file:
            results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (id_batch, number_solved, number_unsolved,
                                                                               planner.size_training_set (),
                                                                               planner.current_budget (), end - start)))
            results_file.write ('\n')

        if number_solved != 0:  # if in the current while loop iteration you solved >= 1 puzzle, then reset planner's budget.
            planner.reset_budget ()  # FD: resets planner's budget to 1, and reset state's budget to 1
            for file, state in states.items ():
                state_budget[state] = 1

            outfile = open (filename_save_ordering, 'ab')
            pickle.dump (solved_puzzles, outfile)
            outfile.close ()

        else:
            planner.increase_budget ()  # FD: increases planner's budget by +1
            continue  # if there are 0 puzzles solved, loop back and try to solve all the puzzles again with an increased

        unsolved_puzzles = current_unsolved_puzzles

        # TODO: on each iteration: save the current weights: we only do this if we know that we are goint to train the NN,
        #  because that means that we will change the weights.
        # TODO: we should store: iteration number (id_batch), weights, puzzles solved in current iteration (P) -- _puzzle_names_ordering_bootstrap,
        #  puzzles solved so far (T), where P in T,
        #  puzzles yet to be solved (S\P).

        # planner.preprocess_data_P ()
        # compute_and_save_cosines(planner._x_P, planner._y_P, nn_model)


        # Note, if no puzzles get solved on iteration i, we increase the budget and loop back. so iteration i might have
        # "sub-iterations" (sub-iteration i1, i2, ..., ik) - maybe new puzzles get added in each sub-iteration.
        # if we have a sub-iteration, it means we did not solve any puzzles and P_new = {}. Also, it means we did not train
        # theta_ik. So, theta_n - theta_ik == theta_n - theta_i and P_new = {}, and thus we do not compute anything.
        # in our log file though, we should save the batch_id,
        # nd under batch_id, the puzzles solved and costs (which indicate the sub-iteration, to some extent). We are already ding this!

        planner.preprocess_data ()
        error = 1
        num_iters = 0
        while num_iters <= gradient_descent_steps:  # while error > epsilon:
            num_iters += 1
            error = planner.learn ()  # this trains the policy? -- if there were 0 solved puzzles, then how does the planner learn?
            print ("id_batch =", id_batch, " num_iters =", num_iters, " error =", error)

        # print ("filepath to save model", join (models_folder, 'model_' + output + "_" + str (id_batch)))
        # # planner.save_model (
        #     # join (models_folder, 'model_' + output + "_" + str (id_batch)))  # FD: "id_batch" is the model id?
        # print("id_batch", id_batch)
        # planner.save_weights (
        #     join (models_folder, 'model_' + output + "_" + str (id_batch) + ".h5"))  # FD: "id_batch" is the model id?

        id_batch += 1
        with open (join (output + '_puzzle_names_ordering_bootstrap'), 'a') as results_file:
            results_file.write (("{:d}".format (id_batch)))
            results_file.write ('\n')
    print ("")
    print("finished test_bootstrap_dfs_lvn_learning_planner")

    planner.save_model (join (models_folder, 'model_' + output + "_" + str(id_batch)))  # FD: "id_batch" is the model id?


def main2():
    if len (sys.argv[1:]) < 1:
        print (
            'Usage for learning a new model: main bootstrap_dfs_lvn_learning_planner <folder-with-puzzles> <output-file> <dropout-rate> <batch-size>')
        print (
            'Usage for using a learned model: main learned_planner <folder-with-puzzles> <output-file> <model-folder>')
        return
    planner_name = sys.argv[1]
    puzzle_folder = sys.argv[2]  # FD: this is the <folder-with-puzzles>

    states = {}  # empty dictionary that will contain the states.
    puzzle_files = [f for f in listdir (puzzle_folder) if
                    isfile (join (puzzle_folder, f))]  # make a list of files in puzzle_folder

    for file in puzzle_files:
        if '.' in file:
            continue  # breaks the current iteration, moves to next iteration of for-loop.
        s = GameState ()  # makes an intance of the class "GameState()"
        s.read_state (
            join (puzzle_folder, file))  # calls method "read_state" and passes the current puzzle_file to this method
        states[file] = s  # adds key = file, value = s to the states dictionary

    KerasManager.register ('KerasModel', KerasModel)
    ncpus = int (os.environ.get ('SLURM_CPUS_PER_TASK', default=1))
    with KerasManager () as manager:
        nn_model = manager.KerasModel ()

        if planner_name == 'bootstrap_dfs_lvn_learning_planner':
            output_file = sys.argv[3]
            dropout = float (sys.argv[4])
            batch = int (sys.argv[5])
            beta = 0.0  # beta is an entropy regularizer term; it isn't currently used in the code

            nn_model.initialize ('CrossEntropyLoss', 'Levin', two_headed_model=False, beta=beta, dropout=dropout)  # we don't give a model folder or model file, b/c no model to loas
            test_bootstrap_dfs_lvn_learning_planner (states, output_file, 1e-1, beta, dropout, batch, nn_model)
            # arguments: states, output, epsilon, beta, dropout, batch

        if planner_name == 'learned_planner':
            output_file = sys.argv[3]
            model_folder = sys.argv[4]
            test_bootstrap_dfs_lvn_learned_model_planner (states, output_file, 0, 1.0, 0, model_folder)
            # FD: inputs are "states, output, beta, dropout, batch, model_folder"
    return


if __name__ == "__main__":
    main2 ()