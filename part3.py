import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import genfromtxt
import copy
import sys

# Global Variables ---
# Nested list of sentences and its words
x = []
# Corresponding labels matching the sentence
y = []
small = sys.float_info.min
label_count = pd.DataFrame
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
            xij,yij = data_pair.split(' ')
            xi.append(xij)
            yi.append(yij)
    return (x,y)
# ---



# To generate lists of X. Splitting each sentence into a seperate sub-list ---
# Input: File Path
# Output: List of x.
def inputGenX(path):

    global x

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

    return x
# ---



# ---
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



# Calculate transition parameters matrix ---
# Input: global variable y will be called
# Output: Transition Parameters matrix
def transitionParameters():
    
    global y

    labels = flatten(y)
    transition_matrix = pd.DataFrame(index = labels, columns = labels)
    transition_matrix.fillna(small,inplace=True)

    for i in range(len(y)):
        for j in range(len(y[i])-1):
            first_word = y[i][j]
            second_word = y[i][j+1]
            transition_matrix.at[str(second_word),str(first_word)] +=1
    for i in labels:
        count = flatten2D(y).count(i)
        transition_matrix[i] = transition_matrix[i].div(count)

    (transition_matrix.sort_index()).to_pickle('transition_matrix')
    return transition_matrix
# ---



# Emission Parameters function
# Input: Global variables x and y will be called
# Output: Emission Parameters matrix
def emissionParameters():
    
    global x
    global y
    global label_count

    words = flatten(x)
    labels = flatten(y)

    emission_matrix = pd.DataFrame(index = words, columns = labels)
    emission_matrix.fillna(small,inplace=True)
    
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        pairs = []
        # Creation of word-label pairs for each sentence
        pairs = list(zip(*[xi,yi]))
        
        for j in pairs:
            # Increment Count(y -> x)
            emission_matrix.at[str(j[0]),str(j[1])] +=1

    # Count(y)
    label_count = emission_matrix.sum(axis=0)
    print(label_count)
    for i in labels:
        # Count(y -> x) / Count(x). Probability of x|y
        emission_matrix[i] = emission_matrix[i].div(label_count[i])
    
    (label_count).to_pickle('label_count')
    (emission_matrix).to_pickle('emission_matrix')
    return (emission_matrix)
# ---



# Get a specific transition probability---
# Input: label and nextlabel (where label -> nextlabel). transition_matrix
# Output: transition probability
def transitionProbability(label, nextlabel, transition_matrix):

    # if (label in transition_matrix.index) and (nextlabel in transition_matrix.columns):
    return transition_matrix.at[label,nextlabel]
    # else:
    #     return 0
# ---  



# Get a specific  emission probability---
# Input: label and word (where label -> word) and k-value. emission_matrix
# Output: emission probability
def emissionProbability(label, word, emission_matrix, k=0.5):
    #print(emission_matrix.index.str.contains("compensation").any())
    
    if not(word in list(emission_matrix.index)):
        return (k / (label_count[label] + k))
    elif word in list(emission_matrix.index):
        return emission_matrix.at[word,label]
# ---



# To produce the labels for Viterbi algo
# Input: x, y, emission matrix and transition matrix
# Output: Sequence
def Viterbi(x,y,em,tm):

    # Number of sentences
    n = len(x)
    sequence = []

    # Initialization of pi matrix
    start = [[1]]
    t_square = [ [ 0 for i in range(len(y)) ] for j in range(n) ]
    stop = [[0]]
    pi = start + t_square + stop

    
    # Perform Viterbi Algo to gather the predicted labels
    for j in range(n):
        pairs = []
        for u in range(0, len(pi[j+1])):
            # Pick maximum score for each entry in pi matrix
            pi[j+1][u] = max([pi[j][v]*emissionProbability(y[u],x[j],em)*transitionProbability(y[v],y[u],tm) for v in range(len(pi[j]))])
            # Gather corresponding label for the maximum score
            pairs.append((u,pi[j+1][u]))
        
        # Conduct backtracking
        index = max(pairs,key=lambda item:item[1])[0]
        sequence.append(y[index])
        print(sequence)

    seq = pd.DataFrame(sequence)
    seq.to_pickle('sequence')
    print(" --- ")
    print(sequence)
    #print(x)
# ---





# import numpy as np

# def fastViterbi(X,Y,em,tm):
#     n = len(X)
#     res_y = []
#     #initialization
#     t_square = np.zeros(shape=(n,len(Y)))

#     for j in range(n):
#         if j == 0:
#             pairs = []
#             for u in range(len(t_square[0])):
#                 t_square[0,u] = np.max([emissionProbability(Y[u],X[j],em)*transitionProbability(Y[0],Y[u],tm)])
#                 pairs.append((u,t_square[0][u]))
#             index = max(pairs,key=lambda item:item[1])[0]
#             res_y.append(Y[index])

#         else:
#             # j = 1 onwards
#             pairs = []
#             for u in range(len(t_square[j])):
#                 t_square[j,u] = np.max([t_square[j-1][v]*emissionProbability(Y[u],X[j],em)*transitionProbability(Y[v],Y[u],tm) for v in range(len(t_square[j-1]))])
#                 pairs.append((u,t_square[j][u]))
#             index = max(pairs,key=lambda item:item[1])[0]
#             res_y.append(Y[index])
#     #print(pi)
#     print(res_y)






# Execution Script --- 
# RUN ONCE FOR FILE CREATION. THEN COMMENT OUT.
x,y = inputGenXY('./Data/EN/train')
# transitionParameters()
# emissionParameters()



# Comment the following for the first run. Then uncomment it for all following runs.
# x = inputGenX('./Data/EN/dev.in')
emission_matrix = pd.read_pickle('emission_matrix')
transition_matrix = pd.read_pickle('transition_matrix')
label_count = pd.read_pickle('label_count')

unique_words =sorted(list(flatten(y)))
unique_words.remove('O')
# Since each sentence starts with a prior state of 'O'
unique_words = ['O'] + unique_words
print('start')
Viterbi(x[30],unique_words,emission_matrix,transition_matrix)


#print(list(emission_matrix.index))
# emissionProbability("B-PP", "compensation", emission_matrix, k=0.5)
# print(len(x))
#y = sorted(list(flatten(y)))
#y = pd.DataFrame(y)
#print(y)
# ---