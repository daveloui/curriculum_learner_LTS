import argparse


def parameter_parser():
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-l', action='store', dest='loss_function',
                         default='CrossEntropyLoss',
                         help='Loss Function')

    parser.add_argument ('-p', action='store', dest='problems_folder', default='problems/witness/puzzles_small/',
                         help='Folder with problem instances')

    parser.add_argument ('-m', action='store', dest='model_name',
                         default="my_NN_model",
                         help='Name of the folder of the neural model')

    parser.add_argument ('-a', action='store', dest='search_algorithm',
                         default='Levin',
                         help='Name of the search algorithm (Levin, LevinStar, AStar, GBFS, PUCT, DFS-Levin)')

    parser.add_argument ('-d', action='store', dest='problem_domain',
                         default='Witness',
                         help='Problem domain (Witness or SlidingTile)')

    parser.add_argument ('-b', action='store', dest='search_budget', default=1,
                         help='The initial budget (nodes expanded) allowed to the bootstrap procedure')

    parser.add_argument ('-g', action='store', dest='gradient_steps', default=10,
                         help='Number of gradient steps to be performed in each iteration of the Bootstrap system')

    parser.add_argument ('-cpuct', action='store', dest='cpuct', default='1.0',
                         help='Constant C used with PUCT.')

    parser.add_argument ('-time', action='store', dest='time_limit', default='43200',
                         help='Time limit in seconds for search')

    parser.add_argument ('-mix', action='store', dest='mix_epsilon', default='0.0',
                         help='Mixture with a uniform policy')

    parser.add_argument ('-parall', action='store', default='1',
                         dest='parallelize_with_NN',
                         help='Are parallelizing any processes that use the NN?')

    parser.add_argument ('-batch_sz', action='store_true', default='1024',
                         dest='batch_size',
                         help='Batch-size for training the NN.')

    parser.add_argument('-diff', action='store', default='final',
                        dest='params_diff',
                        help='Whether we subtract theta_n-theta_i (final) or theta_i+1-theta_i (subsequent) \
                        when computing the debugging data')

    parser.add_argument ('--default-heuristic', action='store_true', default=False,
                         dest='use_heuristic',
                         help='Use the default heuristic as input')

    parser.add_argument ('--learned-heuristic', action='store_true', default=False,
                         dest='use_learned_heuristic',
                         help='Use/learn a heuristic')

    parser.add_argument ('--blind-search', action='store_true', default=False,
                         dest='blind_search',
                         help='Perform blind search')

    parser.add_argument ('--single-test-file', action='store_true', default=False,
                         dest='single_test_file',
                         help='Use this if problem instance is a file containing a single instance.')

    parser.add_argument ('--learn', action='store_true', default=False,
                         dest='learning_mode',
                         help='Train as neural model out of the instances from the problem folder')

    parser.add_argument('--load_debug_data', action='store_true', default=False,
                        dest='load_debug_data',
                        help='Run program and save debug data (cosines, gradients, etc)')

    # new parameters added:
    parser.add_argument ('-dropout', action='store_true', default='1.0',
                         dest='dropout_rate',
                         help='The dropout rate is set to 1.0 (for all methods), which means that no dropout is used. '
                              'Currently, DFS-Levin is the only algorithm which uses dropout.')

    parameters = parser.parse_args ()

    return parameters