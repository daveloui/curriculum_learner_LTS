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

object = open_pickle_file("puzzles_4x4/Rank_MaxDotProd_BFS_4x4.pkl")
print(type(object))
print(len(object[0]))
# print(object)
print("")
object = open_pickle_file("puzzles_4x4/Ordering_BFS_4x4.pkl")
print(type(object))
print(len(object[0]))
print(object)