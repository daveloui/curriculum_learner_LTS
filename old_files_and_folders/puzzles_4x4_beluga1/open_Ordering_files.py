import numpy as np
import pickle


def open_pickle_file(filename):
    objects = []
    with (open (filename, "rb")) as openfile:
        print("filename", filename)
        while True:
            try:
                objects.append (pickle.load (openfile))
            except EOFError:
                break
    openfile.close ()
    print("successfully opened pickle file")
    return objects


array_opened = np.load("Ordering_BFS_4x4.npy", allow_pickle=True)
print("Ordering_BFS_4x4.npy =", array_opened)
print("len(array_opened) =", len(array_opened))
print("")
# p = '3x3_15'
# count_p = 0
# for puzzle in array_opened:
#     if puzzle == p:
#         count_p += 1
# print("count_p", count_p)
#
# object = open_pickle_file("BFS_memory_4x4.pkl")
# print(type(object), " ", len(object))
# print(type(object[0]), len(object[0]))
