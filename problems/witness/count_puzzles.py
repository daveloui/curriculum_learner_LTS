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


def map_witness_puzzles_to_dims(name):
    if (name == "witness_1") or (name == "witness_2"):
        return "1x2"
    elif name == "witness_3":
        return "1x3"
    elif name == "witness_4":
        return "2x2"
    elif (name == "witness_5") or (name == "witness_6"):
        return "3x3"
    elif (name == "witness_7") or (name == "witness_8") or (name == "witness_9"):
        return "4x4"


puzzles_path = os.path.join (os.path.dirname (os.path.realpath (__file__)), "puzzles_4x4/")
print ("puzzle path =", puzzles_path)
# if not os.path.exists (puzzles_path):
#     os.makedirs (puzzles_path, exist_ok=True)


dict = {"1x2": 0, "1x3": 0, "2x2": 0, "3x3": 0, "4x4": 0}
for name in os.listdir(puzzles_path):
    print("name", name)
    if not os.path.isfile(name):
        continue
    if "DS" in name:
        continue

    if "1x2" in name:
        dict["1x2"] += 1
        print(dict)
    elif "1x3" in name:
        dict["1x3"] += 1
    elif "2x2" in name:
        dict["2x2"] += 1
    elif "3x3" in name:
        dict["3x3"] += 1
    elif "4x4" in name:
        dict["4x4"] += 1
    else:
        new_name = map_witness_puzzles_to_dims (name)
        dict[new_name] += 1

print ("dict =", dict)
print ("")