import numpy as np
import os
from pathlib import Path
from collections import Counter, defaultdict
import sys
import pandas as pd
from pandas import DataFrame
import copy



# To generate lists of X and Y. Splitting each sentence into a seperate sub-list ---
# Input: File Path
# Output: List of x. List of y.
def inputGenXY(path):

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
        if (i // k) == (index // k): (j += 1)
# ---


# Retrieve Test Data ---
# Input: File name of test data. Word and their respective index number.
# Output: Nested list of x words. Nested list of x words (index integers).
def InputGenX_2(filename, indexOfWord):

    f = open(path)
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


# Helper function: Initiate key variables
def preProc():
    # Read data to get X, Y
    words, labels = inputGenXY(train_file)
    # Number of time each word occurs. Get each word from each sentence.
    vocab_count = Counter([word for sentence in self.words for word in sentence])
    # List of words that have count >= 7. This is to reduce the number of words that appear too few times.
    vocab = [word for word, count in dict(vocab_count).items() if count >= 7] + ['#UNK#']
    # List of all unique labels with START and STOP // (21 + 2 = 23)
    tag = list(set([t for sentence in labels for t in sentence])) + ['START', 'STOP']
    # Gives index to each label in a Dic
    indexOfTag = {key: value for value, key in enumerate(tag)}
    # A dictionary that stores each word as an index
    indexOfWord = defaultdict(int)
    for i, o in enumerate(self.vocab):
        indexOfWord[o] = i+1
    # Update each x and y value to respective index value
    x = [[indexOfWord[word] for word in sentence] for sentence in words]
    y = [[indexOfTag[word] for word in sentence] for sentence in labels]


class HMM:
    def __init__(self, train_file):
        # Read data
        self.words, self.labels = inputGenXY(train_file)
        # List of all unique labels with START and STOP
        self.tag = list(set([oo for o in self.labels for oo in o])) + ['START', 'STOP']
        # Gives index to each label
        self.indexOfTag = {o: i for i, o in enumerate(self.tag)}
        # Number of time each word occurs. Get each word from each sentence.
        vocab_count = Counter([oo for o in self.words for oo in o])
        # List of words. Barring the first word (special case)
        self.vocab = [o for o, v in dict(
            vocab_count).items() if v >= 3] + ['#UNK#']
        # A dictionary that stores each word as an index
        self.indexOfWord = defaultdict(int)
        for i, o in enumerate(self.vocab):
            self.indexOfWord[o] = i+1
        # Convert each x and y value to respective index value
        self.x = [[self.indexOfWord[oo] for oo in o] for o in self.words]
        self.y = [[self.indexOfTag[oo] for oo in o] for o in self.labels]

    # ---
    # Helper function - returns unique list of elements from d

    def flatten(self, d):
        return {i for b in [[i] if not isinstance(i, list) else self.flatten(i) for i in d] for i in b}

    # Helper function - similar to flatten function but for 2D.

    def flatten2D(self, list2D):

        flatList = []

        for element in list2D:
            if type(element) is list:
                for item in element:
                    flatList.append(item)
            else:
                flatList.append(element)
        return flatList
    # ---

    # Populate the Transition Params Table ---
    # Input: No input. But need to run df first.
    # Output: transParamsTable stored in a pickle
    def transParamsTable(self):

        global transParamsTable

        #rows = ['START']
        rows = []
        columns = []
        for label in self.flatten(y):
            rows.append(label)
            columns.append(label)

        transitionParamsTable = pd.DataFrame(
            index=rows, columns=columns).fillna(0)

        labels = copy.deepcopy(y)
        # labels.append('STOP')
        # labels.insert(0,'START')

        nextLabel = 0

        for i in range(len(labels)):
            for j in range(len(labels[i])-1):
                if nextLabel != 'START':
                    label = labels[i][j]
                    nextLabel = labels[i][j+1]
                    # count(label -> next label)
                    transitionParamsTable.at[label,
                                             nextLabel] = transitionParamsTable.at[label, nextLabel] + 1
                    # print(nextLabel)

        summation = transitionParamsTable.sum()
        # count(label)
        summation = summation.sort_index(ascending=True)
        # print(summation)

        for col in transitionParamsTable.columns:
            # count(label -> next label) / count(label)
            transitionParamsTable[col] = transitionParamsTable[col] / \
                summation[col]

        print("--- Transition Parameters Table populated ---")

        transitionParamsTable = np.log(transitionParamsTable + small)
        (transitionParamsTable.sort_index()).to_pickle('transitionParamsTable')
        print(transitionParamsTable)
        return transitionParamsTable
    # ---

    # Emission Parameters function ---
    # Input: Global variables x and y will be called
    # Output: Emission Parameters matrix
    def emissionParameters(self):

        global x
        global y
        global label_count

        words = self.flatten(x)
        labels = self.flatten(y)

        emission_matrix = pd.DataFrame(index=words, columns=labels)
        emission_matrix.fillna(0, inplace=True)

        for i in range(len(x)):
            xi = x[i]
            yi = y[i]
            pairs = []
            # Creation of word-label pairs for each sentence
            pairs = list(zip(*[xi, yi]))

            for j in pairs:
                # Increment Count(y -> x)
                emission_matrix.at[str(j[0]), str(j[1])] += 1

        # Count(y)
        label_count = emission_matrix.sum(axis=0)
        print(label_count)
        for i in labels:
            # Count(y -> x) / Count(x). Probability of x|y
            emission_matrix[i] = emission_matrix[i].div(label_count[i])

        label_count.to_pickle('label_count')
        emission_matrix = np.log(emission_matrix + small)
        emission_matrix.to_pickle('emission_matrix')
        return emission_matrix
    # ---

    # Get a specific transition probability---
    # Input: label and nextlabel (where label -> nextlabel). transition_matrix
    # Output: transition probability
    def transitionProbability(self, label, nextlabel, transition_matrix):
        return transition_matrix.at[label, nextlabel]
    # ---

    # Get a specific  emission probability ---
    # Input: label and word (where label -> word) and k-value. emission_matrix
    # Output: emission probability
    def emissionProbability(self, label, word, emission_matrix, k=0.5):
        if not(word in word_list):
            return (k / (label_count[label] + k))
        elif word in word_list:
            return emission_matrix.at[word, label]
    # ---

    # Convert our pd Data Frames to Numpy arrays for Part 4 Viterbi ---
    # Input: Dataframe
    # Output: Numpy
    def convertPDtoNP(self, df):
        nump = df.to_numpy()
        return nump
    # ---

    def train(self):
        self.transParamsTable()
        self.emissionParameters()

    def viterbi_top_k(self, x, k=3):
        # Create a 3D numpy array of zeros. For score. + 2 for START STOP and + 1 for #UNK#
        score = np.zeros((len(x)+2, len(self.tag)-2, k))
        # Create a 3D numpy array of zeros with data type integer. For index of argmax.
        argmax = np.zeros((len(x)+2, len(self.tag)-2, k), dtype=np.int)
        # Log trained parameters
        trans = pd.read_pickle('transitionParamsTable')
        em = pd.read_pickle('emission_matrix')
        transition, emission = convertPDtoNP(trans), convertPDtoNP(em)
        print(emission.shape)
        print(transition.shape)
        # Initialization at j=1
        # Populate initial transition and emission values in score table
        score[1, :, 0] = (transition[-1, :-1] + emission[x[0], :-2])
        # Populate score table with - infinity values for all unfilled values
        score[1, :, 1:] = -np.inf
        # j=2, ..., n
        for j in range(2, len(x)+1):
            for t in range(len(self.tag)-2):
                # Load prev word score
                pi = score[j-1, :]  # (num_of_tag-2, 3)
                # Load transmission value
                a = transition[:-1, t]  # (num_of_tag-2,)
                # Load emission value
                b = emission[x[j-1], t]  # (1,)
                previous_all_scores = ((pi + a[:, None])).flatten()
                topk = previous_all_scores.argsort()[-k:][::-1]  # big to small
                # big: 0, small: -1 # topk // k is the real argument
                argmax[j, t] = topk
                # Calculate one plane of scores
                score[j, t] = previous_all_scores[topk] + b

        # j=n+1 step STOP
        pi = score[len(x)]  # (num_of_tag-2, 7)
        a = transition[:-1, -1]
        # big to small
        argmax_stop = (pi + a[:, None]).flatten().argsort()[-k:][::-1]
        log_likelihood = np.min(pi+a[:, None])
        argmax = argmax[2:-1]  # (len(x)-1, num_of_tag-2, 7)

        # decoding
        # Backtracking
        # Initialize the last pointer backward
        temp_index = argmax_stop[-1]  # range from 0 ~ k * ( len(tag) - 1 )
        # range from 0 ~ k-1, next index in next tag
        temp_index_index = getNextIndex(argmax_stop, temp_index, k)
        temp_arg = temp_index // k  # range from 0 ~ k-1, next tag index

        path = [argmax_stop[-1]//k]  # initialize path as an array

        for i in range(len(argmax)-1, -1, -1):
            temp_index = argmax[i, temp_arg, temp_index_index]
            temp_index_index = getNextIndex(
                argmax[i, temp_arg], temp_index, k)
            temp_arg = temp_index // k
            path.append(temp_arg)
        return path[::-1], log_likelihood

    def predict_top_k(self, dev_x_filename, output_filename, k=3):
        with open(output_filename, 'w') as f:
            words, dev_x = InputGenX_2(dev_x_filename, self.indexOfWord)
            score_list = []
            for i, (ws, o) in enumerate(zip(words, dev_x)):
                path, log_max_score = self.viterbi_top_k(o, k)
                score_list.append(log_max_score)
                for w, p in zip(ws, path):
                    f.write(w + ' ' + self.tag[p] + '\n')
                f.write('\n')
        print(score_list)
        return


def convertPDtoNP(df):
    nump = df.to_numpy()
    return nump


AL_dev_x = 'Data/EN/dev.in'
AL_dev_y = 'Data/EN/dev.out'
AL_out_4 = 'dev.p4.out'

hmm = HMM('Data/EN/train')
hmm.train()
# print(hmm.tag)
hmm.predict_top_k(AL_dev_y, AL_out_4, k=3)
# # print("success")
