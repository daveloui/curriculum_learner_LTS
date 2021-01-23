import pickle
import numpy as np
import tensorflow as tf

import time
import sys
import os
from os.path import join
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math

from models.memory import Memory, MemoryV2
from models.model_wrapper import KerasManager, KerasModel
from compute_cosines import compute_cosines, compute_rank, retrieve_final_NN_weights #, save_data_to_disk, find_argmax, compute_levin_cost, find_minimum, compute_rank, compute_rank_mins


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed (1)
tf.random.set_seed (1)
P = ['4x4_979', '4x4_698', '4x4_131', '4x4_609', '4x4_903', '4x4_992', '4x4_488', '4x4_630', '4x4_419', '4x4_229', '4x4_85', '4x4_511', '4x4_580', '4x4_207', '4x4_701', '4x4_281', '4x4_253', '4x4_210', '4x4_344', '4x4_687', '4x4_925', '4x4_642', '4x4_690', '4x4_139', '4x4_966', '4x4_381', '4x4_406', '4x4_154', '4x4_186', '4x4_638', '4x4_143', '4x4_329', '4x4_275', '4x4_803', '4x4_764', '4x4_773', '4x4_262', '4x4_857', '4x4_885', '4x4_588', '4x4_537', '4x4_71', '4x4_879', '4x4_32', '4x4_563', '4x4_709', '4x4_25', '4x4_299', '4x4_35', '4x4_564', '4x4_22', '4x4_869', '4x4_847', '4x4_720', '4x4_598', '4x4_231', '4x4_509', '4x4_763', '4x4_774', '4x4_882', '4x4_490', '4x4_628', '4x4_442', '4x4_107', '4x4_976', '4x4_391', '4x4_300', '4x4_935', '4x4_680', '4x4_317', '4x4_922', '4x4_4', '4x4_386', '4x4_354', '4x4_606', '4x4_780', '4x4_752', '4x4_835', '4x4_200', '4x4_876', '4x4_861', '4x4_797', '4x4_822', '4x4_590', '4x4_95', '4x4_516', '4x4_409', '4x4_904', '4x4_498', '4x4_620', '4x4_372', '4x4_637', '4x4_674', '4x4_326', '4x4_399', '4x4_473', '4x4_308', '4x4_162', '4x4_430', '4x4_175', '4x4_427', '4x4_969', '4x4_464', '4x4_187', '4x4_155', '4x4_116', '4x4_444', '4x4_453', '4x4_481', '4x4_328', '4x4_190', '4x4_617', '4x4_345', '4x4_654', '4x4_306', '4x4_2', '4x4_643', '4x4_924', '4x4_380', '4x4_138', '4x4_352', '4x4_536', '4x4_575', '4x4_24', '4x4_562', '4x4_708', '4x4_521', '4x4_799', '4x4_237', '4x4_220', '4x4_589', '4x4_263', '4x4_337', '4x4_993', '4x4_374', '4x4_363', '4x4_956', '4x4_984', '4x4_631', '4x4_320', '4x4_915', '4x4_198', '4x4_672', '4x4_127', '4x4_130', '4x4_206', '4x4_297', '4x4_245', '4x4_743', '4x4_529', '4x4_791', '4x4_93', '4x4_84', '4x4_228', '4x4_56', '4x4_581', '4x4_553', '4x4_15', '4x4_591', '3x3_290', '4x4_238', '4x4_46', '4x4_51', '4x4_83', '4x4_12', '4x4_753', '4x4_539', '4x4_242', '4x4_877', '4x4_287', '4x4_860', '4x4_216', '4x4_796', '4x4_472', '4x4_618', '3x3_662', '4x4_465', '4x4_968', '4x4_905', '4x4_373', '4x4_119', '4x4_983', '4x4_636', '4x4_327', '4x4_675', '4x4_273', '4x4_227', '4x4_851', '4x4_77', '4x4_209', '4x4_34', '4x4_572', '4x4_565', '4x4_526', '4x4_610', '4x4_681', '4x4_439', '4x4_934', '4x4_644', '4x4_960', '4x4_355', '4x4_387', '4x4_152', '4x4_338', '4x4_180', '4x4_400', '4x4_111', '4x4_491', '4x4_486', '4x4_417', '4x4_112', '4x4_440', '4x4_492', '4x4_669', '4x4_403', '4x4_194', '4x4_988', '4x4_105', '4x4_682', '4x4_168', '4x4_613', '4x4_479', '4x4_974', '4x4_356', '4x4_695', '4x4_6', '4x4_828', '4x4_525', '4x4_761', '4x4_270', '4x4_845', '4x4_852', '4x4_880', '4x4_811', '4x4_945', '4x4_997', '4x4_911', '4x4_324', '4x4_367', '4x4_635', '4x4_432', '4x4_471', '4x4_134', '4x4_425', '4x4_928', '4x4_874', '4x4_579', '4x4_750', '4x4_782', '4x4_820', '4x4_215', '4x4_747', '4x4_284', '4x4_28', '4x4_863', '4x4_769', '4x4_503', '4x4_45', '4x4_278', '4x4_557', '4x4_11', '4x4_80', '4x4_504', '4x4_547', '4x4_582', '4x4_550', '4x4_268', '4x4_779', '4x4_294', '4x4_38', '4x4_246', '4x4_714', '4x4_830', '4x4_757', '4x4_740', '4x4_251', '4x4_283', '4x4_569', '4x4_461', '4x4_359', '4x4_133', '4x4_648', '4x4_377', '4x4_334', '4x4_666', '4x4_916', '4x4_671', '4x4_955', '4x4_360', '4x4_801', '4x4_98', '4x4_725', '4x4_887', '4x4_260', '4x4_816', '4x4_838', '4x4_73', '4x4_64', '4x4_522', '4x4_27', '4x4_685', '4x4_394', '4x4_603', '4x4_383', '4x4_964', '4x4_692', '4x4_640', '4x4_312', '4x4_927', '4x4_998', '4x4_115', '4x4_156', '4x4_193', '4x4_679', '4x4_413', '4x4_450', '4x4_482', '4x4_161', '4x4_135', '4x4_176', '4x4_929', '4x4_449', '4x4_623', '4x4_996', '4x4_332', '4x4_907', '4x4_677', '4x4_910', '4x4_981', '4x4_44', '4x4_96', '4x4_768', '4x4_502', '4x4_593', '4x4_889', '4x4_584', '4x4_81', 'witness_8', '4x4_578', '4x4_240', '4x4_751', '4x4_746', '4x4_794', '4x4_821', '4x4_862', '4x4_975', '4x4_385', '4x4_962', '4x4_357', '4x4_921', '4x4_694', '4x4_441', '4x4_402', '4x4_456', '4x4_104', '4x4_989', '4x4_807', '4x4_232', '4x4_760', '4x4_844', '4x4_549', '4x4_881', '4x4_18', '4x4_853', '4x4_225', '4x4_570', '4x4_248', '4x4_533', '4x4_759', '4x4_75', '4x4_829', '4x4_567', '4x4_839', '4x4_724', '4x4_854', '4x4_261', '4x4_733', '4x4_817', '4x4_446', '4x4_494', '4x4_114', '4x4_999', '4x4_908', '4x4_412', '4x4_140', '4x4_931', '4x4_684', '4x4_395', '4x4_347', '4x4_350', '4x4_602', '4x4_179', '4x4_313', '4x4_641', '4x4_715', '4x4_247', '4x4_204', '4x4_741', '4x4_826', '4x4_568', '4x4_282', '4x4_43', '4x4_594', '4x4_551', '4x4_778', '4x4_512', '4x4_667', '4x4_148', '4x4_423', '4x4_8']

def get_image_representation(state_info_dict):
    # obtains all the state data, returns an image object (more on this later)
    assert state_info_dict['puzzle_type'] == 'Witness'
    number_of_colors = state_info_dict['number_of_colors']
    channels = state_info_dict['channels']

    size = state_info_dict['size']
    columns = size[0]
    lines = size[1]

    startPosition = state_info_dict['startPosition']
    column_init = startPosition[0]
    line_init = startPosition[1]

    maxSize = state_info_dict['maxSize']
    max_columns = maxSize[0]
    max_lines = maxSize[1]

    cells = state_info_dict['cells']
    v_seg = state_info_dict['vertical_seg']
    h_seg = state_info_dict['horizontal_seg']

    tips = state_info_dict['tips']
    column_tip = tips[0]
    line_tip = tips[1]

    endPosition = state_info_dict['endPosition']
    column_goal = endPosition[0]
    line_goal = endPosition[1]

    # defining the 3-dimnesional array that will be filled with the puzzle's information
    image = np.zeros ((2 * max_lines, 2 * max_lines, channels))
    # create one channel for each color i
    for i in range (0, number_of_colors):
        for j in range (0, cells.shape[0]):
            for k in range (0, cells.shape[1]):
                if cells[j][k] == i:
                    image[2 * j + 1][2 * k + 1][i] = 1
    channel_number = number_of_colors

    # the number_of_colors-th channel specifies the open spaces in the grid
    for j in range (0, 2 * lines + 1):
        for k in range (0, 2 * columns + 1):
            image[j][k][channel_number] = 1

    # channel for the current path
    channel_number += 1
    for i in range (0, v_seg.shape[0]):
        for j in range (0, v_seg.shape[1]):
            if v_seg[i][j] == 1:
                image[2 * i][2 * j][channel_number] = 1
                image[2 * i + 1][2 * j][channel_number] = 1
                image[2 * i + 2][2 * j][channel_number] = 1

    for i in range (0, h_seg.shape[0]):
        for j in range (0, h_seg.shape[1]):
            if h_seg[i][j] == 1:
                image[2 * i][2 * j][channel_number] = 1
                image[2 * i][2 * j + 1][channel_number] = 1
                image[2 * i][2 * j + 2][channel_number] = 1

    # channel with the tip of the snake
    channel_number += 1
    image[2 * line_tip][2 * column_tip][channel_number] = 1

    # channel for the exit of the puzzle
    channel_number += 1
    image[2 * line_goal][2 * column_goal][channel_number] = 1

    # channel for the entrance of the puzzle
    channel_number += 1
    image[2 * line_init][2 * column_init][channel_number] = 1

    # print("image shape", image.shape)
    return image


def retrieve_batch_data_solved_puzzles(dictionary, memory_v2):
    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    # all_actions = []

    for puzzle_name, sa_list in dictionary.items():  # puzzles_list:
        print("puzzle_name", puzzle_name)
        states = sa_list[0]
        actions = sa_list[1]
        images_p = []
        actions_one_hot = tf.one_hot (actions, 4)
        # all_actions.append (actions_one_hot)  # a list of tensors
        for state_info_dict in states:
            single_img = get_image_representation (state_info_dict)
            assert isinstance (single_img, np.ndarray)
            images_p.append (single_img)  # a list of np.arrays
        # print("finished inner for loop")

        images_p = np.array (images_p) #, dtype=object)  # convert list of np.arrays to an array
        # images_p = np.asarray (images_p).astype ('float32')
        assert isinstance (images_p, np.ndarray)
        print("images_p.shape", images_p.shape)
        print("actions_one_hot.numpy ().shape", actions_one_hot.numpy ().shape)

        print("now going to store puzzle images")
        memory_v2.store_puzzle_images_actions (puzzle_name, images_p, actions_one_hot.numpy ())  # TODO: to be used by find min
    return  # TODO: the following used to be uncommented: batch_images, batch_actions


def open_pickle_file(filename):
    objects = []
    with (open (filename, "rb")) as openfile:
        while True:
            try:
                objects.append (pickle.load (openfile))
            except EOFError:
                break
    openfile.close ()
    return objects

# flatten_list = lambda l: [item for sublist in l for item in sublist]


def flatten_list(array, dim, shape_tup=None):
    if dim == 1:
        flatten_list_v0 = lambda l: [item for item in l]
        new_list = flatten_list_v0(array)
    elif dim == 2:
        flatten_list_v2 = lambda l: [item for sublist in l for item in sublist]
        new_list = flatten_list_v2 (array)
    elif dim == 3:
        flatten_list_v3 = lambda l: [item for sublist_d0 in l for sublist_d1 in sublist_d0 for item in sublist_d1]
        new_list = flatten_list_v3(array)
    elif dim == 4:
        flatten_list_v4 = lambda l: [item for sublist_d0 in l for sublist_d1 in sublist_d0 for sublist_d2 in sublist_d1 for item in sublist_d2]
        new_list = flatten_list_v4(array)
    else:
        print("Not valid")
    return new_list

# file = "solved_puzzles/puzzles_4x4_theta_n-theta_i/BFS_memory_4x4.pkl"
# object = open_pickle_file(file)
# print("len(obj dict)", len(object[0]))
# print("len", len(list(object[0].values())))
#
# d = {}
# for k, v_list_p in object[0].items():
#     if k in P:
#         states_list = v_list_p[1]
#         actions_list = v_list_p[2]
#         d[k] = [states_list, actions_list]
#         # print("len(v_list_p)", len(v_list_p))
#         # print("number of states", len(states_list))
#         # print("")
# assert P == list(d.keys())


models_folder = 'trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/'
# puzzle_dims = '4x4'
# ordering_folder = 'solved_puzzles/puzzles_' + puzzle_dims  # + "_debug_data"

KerasManager.register ('KerasModel', KerasModel)
with KerasManager () as manager:
    nn_model = manager.KerasModel ()
    nn_model.initialize ("CrossEntropyLoss", "Levin", two_headed_model=False)
    nn_weights_file = "trained_models_large/BreadthFS_4x4-Witness-CrossEntropyLoss/Final_weights_NoDebug.h5"
    nn_model.load_weights (nn_weights_file)
    theta_i = nn_model.retrieve_layer_weights ()

    theta_diff, theta_i_prime, theta_n = compute_cosines (nn_model, models_folder, None, True)

    print("type(theta_diff)", type(theta_diff))
    print("type(theta_i) =", type(theta_i))
    print("type(theta_i_prime)", type(theta_i_prime))
    print("type(theta_n)", type(theta_n))

    list_theta_n = []
    list_theta_i = []
    num_weights_j = 0
    num_weights_i = 0
    for j, sublist_j in enumerate(theta_n):
        sublist_j_array = sublist_j.numpy()
        num_items = np.prod(sublist_j_array.shape)
        dims_j = len (sublist_j_array.shape)
        sublist_j_list = sublist_j_array.tolist()
        flat_j = flatten_list(sublist_j_list, dims_j)
        assert len(flat_j) == num_items

        num_weights_j += len(flat_j)
        list_theta_n.append(flat_j)

        sublist_i_array = theta_i[j].numpy()
        num_items = np.prod (sublist_i_array.shape)
        dims_i = len (sublist_i_array.shape)
        sublist_i_list = sublist_i_array.tolist ()
        flat_i = flatten_list (sublist_i_list, dims_i)
        assert len (flat_i) == num_items
        
        num_weights_i += len(flat_i)
        list_theta_i.append (flat_i)

        assert (flat_j == flat_i)
        # print ("num_items in subarray", num_items)
        # print ("dims of sublist_j_array =", dims_j)
        # print("len(flat_j)", len(flat_j))
        # print ("num_items in subarray", num_items)
        # print ("dims of sublist_i_array =", dims_i)
        # print ("len(flat_i)", len (flat_i))

    # print("num_weights_j =", num_weights_j)
    # print("num_weights_i =", num_weights_i)
    assert num_weights_i == num_weights_j



    # memory_v2 = MemoryV2 (ordering_folder, puzzle_dims, "only_add_solutions_to_new_puzzles")
    # print("")
    # print("calling retrieve_batch_data_solved_puzzles")
    # retrieve_batch_data_solved_puzzles (d, memory_v2)
    #
    # argmax_p_dot_prods, Rank_dot_prods, argmax_p_cosines, Rank_cosines = compute_rank (P, nn_model, theta_diff,
    #                                                                                    memory_v2, 4, 1, 1, False)
    #
    # print("argmax_p_dot", argmax_p_dot_prods)
    # print("Rank_DP", Rank_dot_prods)
    # print("argmax_cosines", argmax_p_cosines)
    # print("Rank cosines", Rank_cosines)





# theta_diff = compute_cosines (nn_model, self._models_folder, while_loop_iter)


# def store_puzzle_images_and_depths(S, T, memory, checkpoint_folder, subfolder_to_save_ordering, parameters, ncpus=None):
#     # make sure that the data is in the format that we expect it to be:
#     if checkpoint_folder is None: # this is only invoked once, at the beginning
#         # print("checkpoint_folder_is_none")
#         for puzzle, values in S.items ():
#             states_list = values[1]
#             actions_list = values[2]
#             actions_one_hot = tf.one_hot (actions_list, 4)  # num_actions = 4
#             actions_one_hot_array = actions_one_hot.numpy()
#             memory.store_labels (actions_one_hot_array, puzzle)
#
#             num_states = len (states_list)
#             assert num_states == len (actions_list)
#             memory.store_log_depths (puzzle, actions_list)
#
#             for i in range (num_states):
#                 state_dict = states_list[i]  # the first state printed is the state preceding the goal state;
#                 memory.store_image (puzzle, state_dict)  # stores image as np.array
#         print("CHECKPOINT FOLDER IS NONE. WE SAVE ALL IMAGES AND LABELS TO SUBFOLDER")
#
#         memory.save_all_images_depths_labels_to_disk(subfolder_to_save_ordering, parameters)
#         all_keys = S.keys()
#     else:
#         # print("checkpoint folder not none")
#         folder_name = 'all_saved_data/images_depths_labels/' + subfolder_to_save_ordering + '/'
#         filename_imgs = os.path.join (folder_name, 'PuzzleImages_PuzzleFolder' + parameters.puzzle_dims + '_run_' +
#                                       parameters.run_idx + '.pkl')
#         with open (filename_imgs, 'rb') as infile:
#             images_dict = pickle.load (infile)
#         infile.close ()
#
#         filename_imgs_saved_so_far = os.path.join (folder_name, 'ImagesAlreadyStored_PuzzleFolder' +
#                                                    parameters.puzzle_dims + '_run_' + parameters.run_idx + '.pkl')
#         with open (filename_imgs_saved_so_far, 'rb') as infile:
#             images_saved_so_far_dict = pickle.load (infile)
#         infile.close ()
#
#         # open labels file:
#         filename_labels = os.path.join (folder_name, 'labels_PuzzleFolder' + parameters.puzzle_dims + '_run_' +
#                                         parameters.run_idx + '.pkl')
#         with open (filename_labels, 'rb') as infile:
#             labels_dict = pickle.load (infile)
#         infile.close ()
#
#         # open depths file:
#         filename_depths = os.path.join (folder_name, 'depths_PuzzleFolder' + parameters.puzzle_dims + '_run_' +
#                                         parameters.run_idx + '.npy')
#         depths_array = np.load(filename_depths, allow_pickle=True)
#
#         memory.populate_memory (images_dict, images_saved_so_far_dict, labels_dict, depths_array.item())
#         S_keys = list (S.keys ())
#         T_keys = list (T.keys ())
#         all_keys = S_keys + T_keys
#
#     print("Inside store_puzzle_images_and_depths. ncpus =", ncpus)
#     if ncpus is not None:
#         array_images_all, array_actions_all, sum_log_depths_all, p_names_all, dict_idxs = prep_batch_images_labels_log_depths (
#             all_keys, memory, int (parameters.chunksize), ncpus)
#         assert list (dict_idxs.keys ()) == p_names_all  # ensure they are in the same order
#         memory.store_dictionary_of_state_action_idxs (dict_idxs)
#         memory.store_batch_training_images_labels (array_images_all, array_actions_all, sum_log_depths_all)
#     return