import numpy as np
import tensorflow as tf
import os
from os.path import join
import pickle
from tensorflow import keras
from tensorflow.keras import layers

from models.model_wrapper import KerasManager, KerasModel
from models.temp_model import TempConvNet #temp_model,

tf.random.set_seed (1)
np.random.seed (1)

# images = [s.get_image_representation () for s in trajectory.get_states ()]
# actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
# _, _, logits = model (np.array (images))


def get_batch_data_all_puzzles(dict):
    # get array of images for each puzzle
    # get array of labels for each puzzle
    # store
    images = []
    for p_name, state in dict.items():
        image = state.get_image_representation ()
        actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
        state.get_image_representation ()
    pass


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


def check_if_data_saved(puzzle_dims):
    batch_folder = 'solved_puzzles/' + puzzle_dims + "/"
    if not os.path.exists (batch_folder):
        os.makedirs (batch_folder, exist_ok=True)
        return False
    # print (os.path.abspath (batch_folder))
    filename = 'Solved_Puzzles_Batch_Data'
    filename = os.path.join (batch_folder, filename + '.pkl')
    return os.path.isfile (filename)


def store_batch_data_solved_puzzles(puzzles_list, memory_v2, puzzle_dims):
    flatten_list = lambda l: [item for sublist in l for item in sublist]

    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    # print("store_or_retrieve_batch_data_solved_puzzles -- len(puzzles_list)", len(puzzles_list))
    all_images_list = []
    all_actions = []
    d = {}
    for puzzle_name in puzzles_list:
        images_p = []
        trajectory_p = memory_v2.trajectories_dict[puzzle_name]
        actions_one_hot = tf.one_hot (trajectory_p.get_actions (), 4)
        all_actions.append(actions_one_hot)   # a list of tensors
        for s in trajectory_p.get_states ():
            single_img = s.get_image_representation ()
            assert isinstance(single_img, np.ndarray)
            images_p.append( single_img ) # a list of np.arrays
        images_p = np.array(images_p, dtype=object)  # convert list of np.arrays to an array
        assert isinstance (images_p, np.ndarray)

        d[puzzle_name] = [images_p, actions_one_hot.numpy()]

        all_images_list.append(images_p) # list of sublists; each sublists contain np.arrays
    assert len(all_images_list) == len(all_actions)


    batch_images = np.vstack(all_images_list)
    batch_actions = np.vstack(all_actions)
    assert batch_images.shape[0] == batch_actions.shape[0]

    # TODO: create folder and file name where we are going to save the dictionary d
    batch_folder = 'solved_puzzles/' + puzzle_dims + "/"
    if not os.path.exists (batch_folder):
        os.makedirs (batch_folder, exist_ok=True)
    filename = 'Solved_Puzzles_Batch_Data'
    filename = os.path.join (batch_folder, filename + '.pkl')
    print("len(d)", len(d))
    outfile = open (filename, 'ab')
    pickle.dump (d, outfile)
    outfile.close ()

    # TODO debug
    # objects = open_pickle_file(filename)
    # print(type(objects[0]))
    # for k, v in objects[0].items():
    #     print("k", k, " v[0].shape", v[0].shape, " v[1].shape", v[1].shape)


def retrieve_batch_data_solved_puzzles(puzzles_list, memory_v2):
    flatten_list = lambda l: [item for sublist in l for item in sublist]

    # TODO: should I save the np.array of images or data that can be used to gnerate the np.arrays?
    # print("store_or_retrieve_batch_data_solved_puzzles -- len(puzzles_list)", len(puzzles_list))
    all_images_list = []
    all_actions = []
    d = {}
    for puzzle_name in puzzles_list:
        images_p = []
        trajectory_p = memory_v2.trajectories_dict[puzzle_name]
        actions_one_hot = tf.one_hot (trajectory_p.get_actions (), 4)
        all_actions.append(actions_one_hot)   # a list of tensors
        for s in trajectory_p.get_states ():
            single_img = s.get_image_representation ()
            assert isinstance(single_img, np.ndarray)
            images_p.append( single_img ) # a list of np.arrays
        images_p = np.array(images_p, dtype=object)  # convert list of np.arrays to an array
        assert isinstance (images_p, np.ndarray)
        d[puzzle_name] = [images_p, actions_one_hot.numpy ()]
        # print("images_p.shape", images_p.shape)
        # print(actions_one_hot.shape)
        # print("")

        all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays
        assert len (all_images_list) == len (all_actions)
        batch_images = np.vstack (all_images_list)
        batch_actions = np.vstack (all_actions)
        assert batch_images.shape[0] == batch_actions.shape[0]

    return batch_images, batch_actions


def retrieve_all_batch_images_actions(d, list_solved_puzzles=None, label="S_minus_P"):
    all_images_list = []
    all_actions = []
    all_images_P = []
    all_actions_P = []
    batch_images = None
    batch_actions = None
    batch_images_P = None
    batch_actions_P = None

    if label == "all_of_S":
        for puzzle_name, v_list in d.items():
            images_p = v_list[0]
            actions_p = v_list[1]
            assert isinstance (images_p, np.ndarray)
            assert isinstance (actions_p, np.ndarray)
            all_actions.append (actions_p)  # a list of tensors
            all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays

    elif label == "P_new":
        for puzzle_name in list_solved_puzzles:
            v_list = d[puzzle_name]
            images_p = v_list[0]
            actions_p = v_list[1]
            assert isinstance (images_p, np.ndarray)
            assert isinstance (actions_p, np.ndarray)
            all_actions_P.append (actions_p)  # a list of tensors
            all_images_P.append (images_p)  # list of sublists; each sublists contain np.arrays

    elif label == "S_minus_P":
        for puzzle_name, v_list in d.items ():
            images_p = v_list[0]
            actions_p = v_list[1]
            assert isinstance (images_p, np.ndarray)
            assert isinstance (actions_p, np.ndarray)

            if puzzle_name in list_solved_puzzles:
                all_actions_P.append (actions_p)  # a list of tensors
                all_images_P.append (images_p)
            else:
                all_actions.append (actions_p)  # a list of tensors
                all_images_list.append (images_p)  # list of sublists; each sublists contain np.arrays

    assert len (all_images_P) == len (all_actions_P)
    if len(all_images_P) > 0:
        batch_images_P = np.vstack (all_images_P)
        batch_actions_P = np.vstack (all_actions_P)
        assert batch_images_P.shape[0] == batch_actions_P.shape[0]
        assert isinstance (batch_images_P, np.ndarray)
        assert isinstance (batch_actions_P, np.ndarray)

    assert len (all_images_list) == len (all_actions)
    if len (all_images_list) > 0:
        batch_images = np.vstack (all_images_list)
        batch_actions = np.vstack (all_actions)
        assert batch_images.shape[0] == batch_actions.shape[0]
        assert isinstance (batch_images, np.ndarray)
        assert isinstance (batch_actions, np.ndarray)

    return batch_images, batch_actions, batch_images_P, batch_actions_P


def get_grads_and_CEL_from_batch(array_images, array_labels, theta_model):
    # print("inside get_grads, using trainset", using_trainset)
    array_images = np.asarray (array_images).astype ('float32')
    array_labels = np.asarray (array_labels).astype ('float32')
    print("passed conversion to array")
    last_grads, mean_loss_val, full_grads = theta_model.get_gradients_from_batch (array_images, array_labels)  # the gradient of the batch (dimensions n x 4)

    num_images = array_images.shape[0]
    sum_loss_val = num_images * mean_loss_val
    return sum_loss_val, mean_loss_val, last_grads

def compute_and_save_cosines_helper_func(theta_diff, last_grads_P):
    pass



def retrieve_last_NN_weights(models_folder, parameters, theta_model, P_batch_images):
    # create toy NN:
    print ("models_folder", models_folder)
    new_model = TempConvNet((2, 2), 32, 4, 'CrossEntropyLoss')
    new_model.load_weights(join (models_folder, "pretrained_weights.h5"))
    print("passed!")
    assert False


def compute_cosines(batch_images_P, batch_actions_P, theta_model, models_folder, parameters):
    print("batch_images_P.shape", batch_images_P.shape)
    print("batch_actions_P.shape", batch_actions_P.shape)

    _, _, last_grads_P = get_grads_and_CEL_from_batch (batch_images_P, batch_actions_P, theta_model)

    print("last_grads_P.shape", last_grads_P.shape)
    print(last_grads_P)
    theta_i = theta_model.retrieve_output_layer_weights()  # shape is (128, 4)
    assert theta_i.shape[-1] == last_grads_P.shape[0]
    print("type(theta_i) =", type(theta_i), " theta_i.shape =", theta_i.shape)

    retrieve_last_NN_weights(models_folder, parameters, theta_model, batch_images_P)

    theta_diff = (theta_n - theta_i)
    compute_and_save_cosines_helper_func (theta_diff, last_grads_P)

    return


def calculate_cosines(batch_images_T, batch_actions_T, log_depths_T, batch_images_S, batch_actions_S, log_depths_S,
                      batch_images_pnew, batch_actions_pnew, log_depths_pnew, theta_model, num_training_set_keys,
                      num_initial_set_keys, memory_model, pnew_still_in_S=False):
    cosine_S_pnew = None
    cosine_S_T = None
    mean_CEL_all_states_T = None
    full_loss_T = None
    mean_full_loss_over_all_puzzles_T = None
    _, mean_CEL_all_states_S, full_loss_S, last_grads_S = get_grads_and_CEL_from_batch (batch_images_S,
                                                                                        batch_actions_S,
                                                                                        log_depths_S,
                                                                                        memory_model,
                                                                                        theta_model, False, True)
    mean_full_loss_over_all_puzzles_S = full_loss_S / num_initial_set_keys

    _, mean_CEL_all_states_pnew, full_loss_pnew, last_grads_pnew = get_grads_and_CEL_from_batch (batch_images_pnew,
                                                                                                 batch_actions_pnew,
                                                                                                 log_depths_pnew,
                                                                                                 memory_model,
                                                                                                 theta_model, False,
                                                                                                 False)
    # online compute cosine_S_pnew if pnew is no longer in S:
    if not pnew_still_in_S:
        dot_prod_pnew_S = np.dot (last_grads_pnew, last_grads_S)
        cosine_S_pnew, _ = compute_cosines (last_grads_pnew, last_grads_S, dot_prod_pnew_S)

    if num_training_set_keys > 0:
        _, mean_CEL_all_states_T, full_loss_T, last_grads_T = get_grads_and_CEL_from_batch (batch_images_T,
                                                                                            batch_actions_T,
                                                                                            log_depths_T,
                                                                                            memory_model,
                                                                                            theta_model,
                                                                                            True, False)
        mean_full_loss_over_all_puzzles_T = full_loss_T / num_training_set_keys
        dot_prod_S_T = np.dot (last_grads_T, last_grads_S)
        cosine_S_T, l2_norm_grad_T = compute_cosines (last_grads_T, last_grads_S, dot_prod_S_T)

    data_to_be_stored = [cosine_S_T, cosine_S_pnew, mean_CEL_all_states_S, full_loss_S, mean_full_loss_over_all_puzzles_S,
                         mean_CEL_all_states_T, full_loss_T, mean_full_loss_over_all_puzzles_T]

    # full_loss_S is the full log loss over all puzzles = sum_full_log_losses
    # sum_loss_S is the sum of -log(pi(s_p)) for all states s_p in S (the sum of cross entropy losses)
    # mean_loss_S is the mean -log(pi(s_p))  " " (the cross entropy losses, which be default return the mean over each state s_p)
    return data_to_be_stored