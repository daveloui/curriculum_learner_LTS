import tensorflow as tf
import numpy as np

from abc import ABC
import math

tf.random.set_seed (1)
np.random.seed (1)


class LossFunction (ABC):

    def compute_loss(self, trajectory, model):
        pass


class ImprovedLevinLoss (LossFunction):

    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation () for s in trajectory.get_states ()]
        actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
        _, _, logits = model (np.array (images))

        weights = model.get_weights ()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm (w, ord=2)

        loss = self.cross_entropy_loss (actions_one_hot, logits)

        d = len (trajectory.get_actions ()) + 1
        pi = trajectory.get_solution_pi ()
        expanded = trajectory.get_non_normalized_expanded () + 1

        a = 0
        if pi < 1.0:
            a = (math.log ((d + 1) / expanded)) / math.log (pi)
        if a < 0:
            a = 0

        loss *= tf.stop_gradient (tf.convert_to_tensor (expanded * a, dtype=tf.float64))

        loss += model._reg_const * weights_l2_norm

        return loss


class RegLevinLoss (LossFunction):

    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation () for s in trajectory.get_states ()]
        actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
        _, probs_softmax, _ = model (np.array (images))

        probs_used_on_path = tf.math.multiply (tf.cast (actions_one_hot, dtype=tf.float64), probs_softmax)
        probs_used_on_path = tf.math.reduce_sum (probs_used_on_path, axis=1)
        solution_prob = tf.math.reduce_prod (probs_used_on_path)

        weights = model.get_weights ()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm (w, ord=2)

        solution_costs = trajectory.get_solution_costs ()
        solution_cost = solution_costs[len (solution_costs) - 1]

        loss = tf.math.divide (tf.convert_to_tensor (solution_cost, dtype=tf.float64), solution_prob)
        loss += model._reg_const * weights_l2_norm

        return loss


class LevinLoss (LossFunction):

    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation () for s in trajectory.get_states ()]
        actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
        _, _, logits = model (np.array (images))

        weights = model.get_weights ()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm (w, ord=2)

        loss = self.cross_entropy_loss (actions_one_hot, logits)

        #         loss *= tf.stop_gradient(tf.convert_to_tensor(trajectory.get_expanded(), dtype=tf.float64))
        loss *= tf.stop_gradient (tf.convert_to_tensor (trajectory.get_non_normalized_expanded (), dtype=tf.float64))
        loss += model._reg_const * weights_l2_norm

        return loss


class CrossEntropyLoss (LossFunction):

    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy (from_logits=True)

    def compute_loss_w_batch(self, batch_actions, preds):
        return self.cross_entropy_loss (batch_actions, preds)

    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation () for s in trajectory.get_states ()]
        actions_one_hot = tf.one_hot (trajectory.get_actions (), model.get_number_actions ())
        array_images = np.array (images)
        # print("puzzles_small =========")
        # # x_log_softmax, x_softmax, logits = model.call (array_images)
        # # print(x_softmax)
        # # print(actions_one_hot)
        # # assert False
        # print("end puzzles_small ========")
        _, _, logits = model.call (array_images)

        weights = model.get_weights ()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm (w, ord=2)

        return self.cross_entropy_loss (actions_one_hot, logits) #+ model._reg_const * weights_l2_norm