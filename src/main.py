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

from game_state import GameState
from bootstrap_dfs_learning_planner import BootstrapDFSLearningPlanner
from parameter_parser import parameter_parser

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


def main():
    """
    It is possible to use this system to either train a new neural network model through the bootstrap system and
    Levin tree search (LTS) algorithm, or to use a trained neural network with LTS.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parameters = parameter_parser()


    states = {}

    if parameters.problem_domain == 'SlidingTile' and parameters.single_test_file:
        with open (parameters.problems_folder, 'r') as file:
            problem = file.readline ()
            instance_name = parameters.problems_folder[
                            parameters.problems_folder.rfind ('/') + 1:len (parameters.problems_folder)]
            puzzle = SlidingTilePuzzle (problem)
            states[instance_name] = puzzle

    elif parameters.problem_domain == 'SlidingTile':
        puzzle_files = [f for f in listdir (parameters.problems_folder) if
                        isfile (join (parameters.problems_folder, f))]

        for filename in puzzle_files:
            with open (join (parameters.problems_folder, filename), 'r') as file:
                problems = file.readlines ()

                for i in range (len (problems)):
                    puzzle = SlidingTilePuzzle (problems[i])
                    states['puzzle_' + str (i)] = puzzle

    elif parameters.problem_domain == 'Witness':
        puzzle_files = [f for f in listdir (parameters.problems_folder) if
                        isfile (join (parameters.problems_folder, f))]

        for file in puzzle_files:
            if '.' in file:
                continue
            s = WitnessState ()
            s.read_state (join (parameters.problems_folder, file))
            states[file] = s
    elif parameters.problem_domain == 'Sokoban':
        problem = []
        puzzle_files = []
        if isfile (parameters.problems_folder):
            puzzle_files.append (parameters.problems_folder)
        else:
            puzzle_files = [join (parameters.problems_folder, f) for f in listdir (parameters.problems_folder) if
                            isfile (join (parameters.problems_folder, f))]

        problem_id = 0

        for filename in puzzle_files:
            with open (filename, 'r') as file:
                all_problems = file.readlines ()

            for line_in_problem in all_problems:
                if ';' in line_in_problem:
                    if len (problem) > 0:
                        puzzle = Sokoban (problem)
                        states['puzzle_' + str (problem_id)] = puzzle

                    problem = []
                    #                 problem_id = line_in_problem.split(' ')[1].split('\n')[0]
                    problem_id += 1

                elif '\n' != line_in_problem:
                    problem.append (line_in_problem.split ('\n')[0])

            if len (problem) > 0:
                puzzle = Sokoban (problem)
                states['puzzle_' + str (problem_id)] = puzzle

    print ('Loaded ', len (states), ' instances')
    #     input_size = s.get_image_representation().shape

    #     set_start_method('forkserver', force=True)

    KerasManager.register ('KerasModel', KerasModel)
    ncpus = int (os.environ.get ('SLURM_CPUS_PER_TASK', default=1))

    k_expansions = 32

    #     print('Number of cpus available: ', ncpus)

    with KerasManager () as manager:

        nn_model = manager.KerasModel ()
        # If you are not loading data, you create the nn_folder.
        # Else, if you are loading data from a previous experiment, checkpoint_folder == 'path_to_save_ordering_and_model'
        if parameters.checkpoint:
            nn_folder = "trained_models_large/Breadth_FS" + parameters.puzzle_dims + "-" + parameters.model_name
            path_to_retrieve_NN = os.path.abspath (nn_folder)
            print(path_to_retrieve_NN)
            new_model.load_weights (join (path_to_retrieve_NN, "checkpointed_weights.h5"))
            print("passed getting NN weight checkpoint")
        # TODO: when you are or are not loading files, save sets T and S (or names of puzzles in T and S), or positions of these puzzles

        bootstrap = None

        if parameters.learning_mode:
            bootstrap = Bootstrap (states, parameters.model_name,
                                   parameters.scheduler, bool(int(parameters.use_epsilon)),
                                   ncpus=ncpus,
                                   initial_budget=int (parameters.search_budget),
                                   gradient_steps=int (parameters.gradient_steps))

        if parameters.search_algorithm == 'PUCT':

            bfs_planner = PUCT (parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions,
                                float (parameters.cpuct))

            if parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=True)
            else:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=False)

            if parameters.learning_mode:
                # bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems (bfs_planner, nn_model)
            elif parameters.blind_search:
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))
            else:
                nn_model.load_weights (join ('trained_models_large', parameters.model_name, 'model_weights'))
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))

        if parameters.search_algorithm == 'Levin' or parameters.search_algorithm == 'LevinStar':

            if parameters.search_algorithm == 'Levin':
                bfs_planner = BFSLevin (parameters.use_heuristic, parameters.use_learned_heuristic, False, k_expansions,
                                        float (parameters.mix_epsilon))
            else:
                bfs_planner = BFSLevin (parameters.use_heuristic, parameters.use_learned_heuristic, True, k_expansions,
                                        float (parameters.mix_epsilon))

            if parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=True)
            else:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=False)

            if parameters.learning_mode:
                #                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems (bfs_planner, nn_model, parameters)
            elif parameters.blind_search:
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))
            else:
                nn_model.load_weights (join ('trained_models_large', parameters.model_name, 'model_weights'))
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))

        if parameters.search_algorithm == 'LevinMult':

            bfs_planner = BFSLevinMult (parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)

            if parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=True)
            else:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=False)

            if parameters.learning_mode:
                #                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems (bfs_planner, nn_model)
            elif parameters.blind_search:
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))
            else:
                nn_model.load_weights (join ('trained_models_large', parameters.model_name, 'model_weights'))
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))

        if parameters.search_algorithm == 'AStar':
            bfs_planner = AStar (parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)

            if parameters.learning_mode and parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm)
                #                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems (bfs_planner, nn_model)
            elif parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm)
                nn_model.load_weights (join ('trained_models_large', parameters.model_name, 'model_weights'))
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))
            else:
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))

        if parameters.search_algorithm == 'GBFS':
            bfs_planner = GBFS (parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)

            if parameters.learning_mode:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm)
                #                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems (bfs_planner, nn_model)
            elif parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm)
                nn_model.load_weights (join ('trained_models_large', parameters.model_name, 'model_weights'))
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))
            else:
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))


if __name__ == "__main__":
    main()
    # main2 ()