import sys
import os
import os.path as path
from os import listdir
from os.path import isfile, join

import argparse
import pickle
import copy
import random
import math
from typing import Dict, Any
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from Witness_Puzzle_Image import WitnessPuzzle


def open_pickle_file(filename):
    objects = []
    with (open (filename, "rb")) as openfile:
        # print("filename", filename)
        while True:
            try:
                objects.append (pickle.load (openfile))
            except EOFError:
                break
    openfile.close ()
    # print("passed")
    # print("")
    return objects

object = open_pickle_file("4x4/cosine_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(len(object[0]))
print(object)
sum_over_data = sum(object[0])
print(sum_over_data)
print("")
object = open_pickle_file("4x4/dot_prod_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(len(object[0]))
print(object)
print("")
print("levin_cost_data")
object = open_pickle_file("4x4/levin_cost_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(len(object[0]))
print(object)
print("")
print("aver_levin_cost_data")
object = open_pickle_file("4x4/aver_levin_cost_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(len(object[0]))
print(object)
print("")
print("training_loss_data")
object = open_pickle_file("4x4/training_loss_data_4x4-Witness-CrossEntropyLoss.pkl")
print(type(object))
print(len(object[0]))
print(object)

