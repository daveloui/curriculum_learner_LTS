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

from bootstrap import Bootstrap
from multiprocessing import set_start_method

from game_state import GameState
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
    import tensorflow as tf
    # print ("inside main")
    # if tf.test.gpu_device_name ():
    #     print ('Default GPU Device:{}'.format (tf.test.gpu_device_name ()))
    # else:
    #     print ("Please install GPU version of TF")

    physical_devices = tf.config.experimental.list_physical_devices ('GPU')
    try:
        tf.config.experimental.set_memory_growth (physical_devices[0], True)
        print ("passed -- tf.config.experimental.set_memory_growth")
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print ("did not pass -- tf.config.experimental.set_memory_growth")
        pass


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parameters = parameter_parser()

    states = {}

    if parameters.problem_domain == 'Witness':
        puzzle_files = [f for f in listdir (parameters.problems_folder) if
                        isfile (join (parameters.problems_folder, f))]

        for file in puzzle_files:
            if '.' in file:
                continue
            s = WitnessState ()
            s.read_state (join (parameters.problems_folder, file))
            states[file] = s

    print ('Loaded ', len (states), ' instances')
    #     input_size = s.get_image_representation().shape
    #     set_start_method('forkserver', force=True)

    KerasManager.register ('KerasModel', KerasModel)
    ncpus = int (os.environ.get ('SLURM_CPUS_PER_TASK', default=1))

    k_expansions = 32

    #     print('Number of cpus available: ', ncpus)

    with KerasManager () as manager:

        nn_model = manager.KerasModel ()
        print("after nn_model = manager.KerasModel ()")

        bootstrap = None

        if parameters.learning_mode:
            bootstrap = Bootstrap (states, parameters.model_name,
                                   parameters.scheduler, bool(int(parameters.use_epsilon)),
                                   ncpus=ncpus,
                                   initial_budget=int (parameters.search_budget),
                                   gradient_steps=int (parameters.gradient_steps))
            print("created Bootstrap")

        if parameters.search_algorithm == 'Levin' or parameters.search_algorithm == 'LevinStar':

            if parameters.search_algorithm == 'Levin':
                print("parameters.search_algorithm == Levin" is True)
                bfs_planner = BFSLevin (parameters.use_heuristic, parameters.use_learned_heuristic, False, k_expansions,
                                        float (parameters.mix_epsilon))
                print("created an instance bfs_planner")
            else:
                bfs_planner = BFSLevin (parameters.use_heuristic, parameters.use_learned_heuristic, True, k_expansions,
                                        float (parameters.mix_epsilon))

            if parameters.use_learned_heuristic:
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=True)
            else:
                print("going to initialize NN")
                nn_model.initialize (parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
                print("finished initializing NNs")

            # If you are not loading data, you create the nn_folder.
            # Else, if you are loading data from a previous experiment, checkpoint_folder == 'path_to_save_ordering_and_model'
            if parameters.checkpoint:
                # puzzle_dims = parameters.model_name.split ('-')[0]
                print ("")
                print ("parameters.use_epsilon", parameters.use_epsilon)
                print ("")

                nn_folder = 'trained_models_large/BreadthFS_' + parameters.model_name + "_" + "use_epsilon=" + str (
                    bool (int (parameters.use_epsilon)))
                path_to_retrieve_NN = os.path.abspath (nn_folder)
                print ("path_to_retrieve_NN =", path_to_retrieve_NN)
                nn_model.load_weights (join (path_to_retrieve_NN, "checkpointed_weights.h5"))
                print ("passed getting NN weight checkpoint")
            # TODO: when you are or are not loading files, save sets T and S (or names of puzzles in T and S), or positions of these puzzles

            if parameters.learning_mode:
                print ("parameters.learning_mode is True")
                solve_problems_start_time = time.time()
                bootstrap.solve_problems (bfs_planner, nn_model, parameters)
                solve_problems_end_time = time.time()
                print("time to execute entire prog =", solve_problems_end_time - solve_problems_start_time)
                print ("finished executing bootstrap.solve_problems (bfs_planner, nn_model)")
            elif parameters.blind_search:
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))
            else:
                nn_model.load_weights (join ('trained_models_large', parameters.model_name, 'model_weights'))
                search (states, bfs_planner, nn_model, ncpus, int (parameters.time_limit),
                        int (parameters.search_budget))




if __name__ == "__main__":
    main()
    # main2 ()