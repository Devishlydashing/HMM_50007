import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict
import sys
import pandas as pd
from pandas import DataFrame
import copy



# Global Variables ---
# Nested list of sentences and its words
x = []
# Corresponding labels matching the sentence
y = []
# Smallest value possible
small = sys.float_info.min
words = []
labels = []
word_count = []
ls_words = []
tag = []
indexOfTag = {}
indexOfWord = {}
emission = []
transition = []

# ---




# To generate lists of X and Y. Splitting each sentence into a seperate sub-list ---
# Input: File Path
# Output: List of x. List of y.
def inputGenXY(path):

    global x
    global y

    f = open(path)
    f_content = f.read()
    # Nested list of sentences and its words
    x = []
    # Corresponding labels matching the sentence
    y = []
    xi = []
    yi = []

    for data_pair in f_content.split('\n'):
        if data_pair == '':
            if (xi != []):
                x.append(xi)
                y.append(yi)
                xi = []
                yi = []
        else:
           # print(data_pair)
            yij = data_pair.split(" ")[-1]
            xij = data_pair.strip(" "+yij)
            xi.append(xij)
            yi.append(yij)

    return x, y
# ---


# Get next index in next tag ---
# Input: Argmax, index, kth best
# Output: Index
def getNextIndex(argmax, index, k):
    j = 0
    for i in argmax:
        if (i == index): return j
        if (i // k) == (index // k): 
            j += 1
# ---


# Retrieve Test Data ---
# Input: File name of test data. Word and their respective index number.
# Output: Nested list of x words. Nested list of x words (index integers).
def InputGenX_2(filename, indexOfWord):

    global x

    f = open(filename)
    f_content = f.read()
    # Nested list of sentences and its words
    x = []
    xi = []
    for data in f_content.split('\n'):
        if data == '':
            # Sentence completed
            if (xi != []):
                x.append(xi)
                xi = []
        else:
            xi.append(data)

    # Convert list of x with respective index number
    x_indx = [[indexOfWord[word] for word in sentence] for sentence in x]
    return x, x_indx
# ---


# Helper function - Initiate key variables
def preProc(training_file):

    global x
    global y
    global words
    global labels
    global word_count
    global ls_words
    global tag
    global indexOfTag
    global indexOfWord

    # Read data to get X, Y
    words, labels = inputGenXY(training_file)
    # Number of time each word occurs. Get each word from each sentence.
    word_count = Counter([word for sentence in words for word in sentence])
    # List of words that have count >= 7. This is to reduce the number of words that appear too few times.
    ls_words = [word for word, count in dict(word_count).items() if count >= 7] + ['#UNK#']
    # List of all unique labels with START and STOP // (21 + 2 = 23)
    tag = list(set([t for sentence in labels for t in sentence])) + ['START', 'STOP']
    # Gives index to each label in a Dic
    indexOfTag = {key: value for value, key in enumerate(tag)}
    # A dictionary that stores each word as an index
    indexOfWord = defaultdict(int)
    for i, o in enumerate(ls_words):
        indexOfWord[o] = i+1
    # Update each x and y value to respective index value
    x = [[indexOfWord[word] for word in sentence] for sentence in words]
    y = [[indexOfTag[word] for word in sentence] for sentence in labels]

# Helper function - returns unique list of elements from d
def flatten(d):
    return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}

# Helper function - similar to flatten function but for 2D.
def flatten2D(list2D):

    flatList = []

    for element in list2D:
        if type(element) is list:
            for item in element:
                flatList.append(item)
        else:
            flatList.append(element)
    return flatList
# ---



# Emission Parameters function ---
# Input: Global variables x and y will be called
# Output: Emission Parameters matrix
def emissionParamsTable(trainFile):

    global emission
    global x,y
    # # Input training file
    # x, y = inputGenXY(trainFile)

    # Initialise an array of zeros. For Count(u -> o)
    emission = np.zeros((len(ls_words), len(tag)))
    # flat_x = flatten(x)
    # flat_y = flatten(y)

    flat_y = [j for i in y for j in i]
    flat_x = [j for i in x for j in i]

    for xi, yi in zip(flat_x, flat_y):
        emission[xi, yi] += 1

    # Initialise an array of zeros. For Count(u)
    y_count = np.zeros(len(tag))
    for yi in flat_y:
        y_count[yi] += 1

    # Count(u -> o) / Count(u)
    emission = emission / y_count[None, :]
    np.nan_to_num(emission, 0)
    # Smoothening and Log
    emission = np.log(emission + sys.float_info.min)
    return emission
# ---



# Populate the Transition Params Table ---
# Input: No input. But need to run df first.
# Output: Will return transition matrix
def transitionParamsTable(trainFile):

    global transition
    global x 
    global y

    # # Input training file
    # x, y = inputGenXY(trainFile)

    # Initialise an square array of zeros.
    transition = np.zeros((len(tag)-1, len(tag)-1))
    
    for yi in y:
        # START
        transition[-1, yi[0]] += 1  
        # From 0 to n-1
        for i in range(len(yi)-1):  # tags transition from position 0 to len(yi)-2
            # Count(u -> v)
            transition[yi[i], yi[i+1]] += 1
        transition[yi[-1], -1] += 1  # STOP transition
    # Count(u -> v) / Count(u)
    transition = transition/np.sum(transition, axis=1)
    # Smoothening and Log
    transition = np.log(transition + sys.float_info.min)
    return transition
# ---


def viterbiK(x, transition, emission, k=3):

    global tag

    # Create a 3D numpy array of zeros. For score. + 2 for START STOP and + 1 for #UNK#
    score = np.zeros((len(x)+2, len(tag)-2, k))
    # Create a 3D numpy array of zeros with data type integer. For index of argmax.
    argmax = np.zeros((len(x)+2, len(tag)-2, k), dtype=np.int)
    
    
    # Initialization at j=1 START
    # Populate initial transition and emission values in score table
    score[1, :, 0] = (transition[-1, :-1] + emission[x[0], :-2])
    # Populate score table with - infinity values for all unfilled values
    score[1, :, 1:] = -np.inf
    
    # j=2, ..., n
    for j in range(2, len(x)+1):
        for t in range(len(tag)-2):
            # Load prev word score
            pi = score[j-1, :]
            # Load transition value
            a = transition[:-1, t]
            # Load emission value
            b = emission[x[j-1], t]
            previous_all_scores = ((pi + a[:, None])).flatten()
            topk = previous_all_scores.argsort()[-k:][::-1]
            argmax[j, t] = topk
            # Calculate one plane of scores
            score[j, t] = previous_all_scores[topk] + b

    # j=n+1 step STOP
    pi = score[len(x)]  # (num_of_tag-2, 7)
    a = transition[:-1, -1]
    # Arrange in descending order
    arg_stop = (pi + a[:, None]).flatten().argsort()[-k:][::-1]
    logLikelihood = np.min(pi+a[:, None])
    argmax = argmax[2:-1]


    # Backtracking
    # Initialize the last pointer backward as a temp index
    t_i = arg_stop[-1]
    t_next_i = getNextIndex(arg_stop, t_i, k)
    temp_arg = t_i // k
    # Init the sequence as an array
    sequence = [arg_stop[-1]//k]

    for i in range(len(argmax)-1, -1, -1):
        t_i = argmax[i, temp_arg, t_next_i]
        t_next_i = getNextIndex(
            argmax[i, temp_arg], t_i, k)
        temp_arg = t_i // k
        sequence.append(temp_arg)
    return sequence[::-1], logLikelihood

def executeViterbiK(trainingFile, devIn, filepath, k=3):
    
    global transition
    global emission

    # Load Trained Params
    transition, emission = transitionParamsTable(trainingFile), emissionParamsTable(trainingFile)
    
    # Store results specified output file
    with open(filepath, 'w') as f:
        words, x_in = InputGenX_2(devIn, indexOfWord)
        scores = []
        for i, (ws, x) in enumerate(zip(words, x_in)):
            # Run modified Viterbi
            seq, maxScore = viterbiK(x, transition, emission, k=3)
            scores.append(maxScore)
            for w, p in zip(ws, seq):
                f.write(w + ' ' + tag[p] + '\n')
            f.write('\n')
    f.close()
    return None




preProc('./Data/EN/train')
executeViterbiK('./Data/EN/train', './Data/EN/dev.in', './Data/EN/dev.p4.out', k=3)