from __future__ import division
import sys
import os
import argparse

ARGS = dict()
NAMES_INDEX = dict()
MATRIX = None

#  From 0 to N_CLASSES
N_CLASSES = 0

#  Access with class-1
CLASS_SIZE = []


def UI():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Directory path",
                        required=True)
    ARGS = vars(parser.parse_args())


def load():
    global NAMES_INDEX, MATRIX, N_CLASSES, CLASS_SIZE
    FOLDER_PATH = ARGS["directory"]
    curr_class = 0
    FILES = os.listdir(FOLDER_PATH)
    FILES.sort()
    for f in FILES:
        if f != ".DS_Store":
            curr_id = getImageID(f)
            curr_class = getImageClass(f)
            N_CLASSES = max(curr_class, N_CLASSES)
            #if curr_id > 2:
                #continue
            if NAMES_INDEX.get(curr_id) is None:
                NAMES_INDEX[curr_id] = []
            NAMES_INDEX[curr_id].append(f)

    CLASS_SIZE = [0 for i in range(N_CLASSES)]

    MATRIX = []
    for i in range(N_CLASSES):
        MATRIX.append([])
        for j in range(N_CLASSES):
            MATRIX[i].append(0)


def printMatrix():
    output = open("matrix.txt", "w")
    string = ''
    for i in range(N_CLASSES+1):
        if i == 0:
            string = "\t\t"
        else:
            string += str(i) + "\t\t"
    print string
    output.write(string+"\n")
    string = ''
    for i in range(N_CLASSES):
        string = str(i+1) + "\t\t"
        for j in range(N_CLASSES):
            if i == j:
                string += "X" + "\t\t"
            else:
                string += "%.3fp \t\t" % (MATRIX[i][j])
        print string
        output.write(string+"\n")
    print "\n\n"
    output.close()


def compute():
    for img in NAMES_INDEX:
        curr_names = NAMES_INDEX[img]
        for i, name in enumerate(curr_names):
            curr_class_i = getImageClass(name)
            curr_id_i = getImageID(name)
            curr_set_i = set()
            content = readImage(name)
            for pos, val in enumerate(content):
                #  If it is an object pixel
                if val == 1:
                    curr_set_i.add(pos)
            size_i = len(curr_set_i)

            j = i+1
            for name_j in curr_names[i+1:]:
                print "Testing: ", name, "with", name_j
                intersection_size, size_j = intersection(curr_set_i, name_j)
                MATRIX[i][j] = max(MATRIX[i][j], float((intersection_size/size_i)*100))
                MATRIX[j][i] = max(MATRIX[j][i], float((intersection_size/size_j)*100))
                j += 1


def intersection(set1, name2):
    curr_class_j = getImageClass(name2)
    curr_id_j = getImageID(name2)
    curr_set_j = set()
    content = readImage(name2)
    counter = 0
    for pos, val in enumerate(content):
        #  If it is an object pixel
        if val == 1:
            counter+=1
            curr_set_j.add(pos)

    return len(set1.intersection(curr_set_j)), counter


def readImage(name):
    fp = open(ARGS["directory"]+name)
    content = fp.read()
    fp.close()
    content = content.split(" ")[:-1]
    content = map(int, content)
    return content


def getImageID(name):
    prefix = name.split(".")[0]
    return int(prefix.split("_")[1])


def getImageClass(name):
    return int(name.split("_")[0])


def main():
    UI()
    load()
    compute()
    printMatrix()


main()
