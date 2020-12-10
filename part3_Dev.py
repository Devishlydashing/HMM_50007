import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import genfromtxt
import copy
import sys
import time

# Global Variables ---
# Nested list of sentences and its words
x = []
# Corresponding labels matching the sentence
y = []
small = sys.float_info.min
label_count = pd.DataFrame
word_list = []
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
            xij,yij = data_pair.split(" ")
            xi.append(xij)
            yi.append(yij)

    #y = sorted(list(flatten(y)))
    #y = pd.DataFrame(y)
    #y.to_pickle('y')

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
    transition_matrix.fillna(0,inplace=True)

    for i in range(len(y)):
        for j in range(len(y[i])-1):
            first_word = y[i][j]
            second_word = y[i][j+1]
            transition_matrix.at[str(second_word),str(first_word)] +=1
    for i in labels:
        count = flatten2D(y).count(i)
        transition_matrix[i] = transition_matrix[i].div(count)

    transition_matrix = np.log(transition_matrix.sort_index() + small)
    (transition_matrix).to_pickle('transition_matrix')
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
    emission_matrix.fillna(0,inplace=True)
    
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
    
    label_count.to_pickle('label_count')
    emission_matrix = np.log(emission_matrix + small)
    emission_matrix.to_pickle('emission_matrix')
    return emission_matrix
# ---



# Get a specific transition probability---
# Input: label and nextlabel (where label -> nextlabel). transition_matrix
# Output: transition probability
def transitionProbability(label, nextlabel, transition_matrix):
    return transition_matrix.at[label,nextlabel]
# ---  



# Get a specific  emission probability---
# Input: label and word (where label -> word) and k-value. emission_matrix
# Output: emission probability
def emissionProbability(label, word, emission_matrix, k=0.5):

    if not(word in word_list):
        return (k / (label_count[label] + k))
    elif word in word_list:
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
        # For each new sentence start a fresh pi
        pairs = []
        for u in range(0, len(pi[j+1])):
            # Pick maximum score for each entry in pi matrix
            pi[j+1][u] = max([pi[j][v]*emissionProbability(y[u],x[j],em)*transitionProbability(y[v],y[u],tm) for v in range(len(pi[j]))])
            # Gather corresponding label and score for the maximum score
            pairs.append((u,pi[j+1][u]))
        
        # Conduct backtracking
        #print(pairs)
        index = max(pairs,key=lambda item:item[1])[0]
        sequence.append(y[index])

    
    #print(" --- ")
    #print(sequence)
    return sequence
# ---




import numpy as np

def simpleViterbi(sentence,labels,em,tm):
    n = len(sentence)
    #last word of sentence must be a fullstop
    startword = [[1]]
    #stopword = [[0]]
    #body = [[0]*len(labels)]*len(sentence)
    body = [ [ 0 for i in range(len(labels)) ] for j in range(len(sentence)) ]
    pi = startword+body
    sentence = ['START'] + sentence

    for i in range(1,len(sentence)):
        for v in range(len(pi[i])):
            max_val = 0
            for u in range(len(pi[i-1])):
                val = pi[i-1][u] * transitionProbability(labels[u],labels[v],tm) * emissionProbability(labels[v],sentence[i],em)
                if val > max_val:
                    max_val = val

            pi[i][v] = max_val
    
    #backtracking
    res_y = [labels[np.argmax(pi[n-1])]]
    for j in range(n-2,-1,-1):
        maxval = 0
        label = labels[0]
        for v in range(len(pi[j])):
            val = pi[j][v]*transitionProbability(labels[v],res_y[n-j-2],tm)
            if val > maxval:
                maxval = val
                label = labels[v]
        res_y.append(label)
    return res_y


def logViterbi(X,Y,em,tm):
    n = len(X)
    res_y = []
    #initialization
    start = [[1]]
    t_square = [ [ 0 for i in range(len(Y)) ] for j in range(n) ]
    stop = [[0]]
    pi = start + t_square + stop
    #perform Viterbi
    for j in range(n):
        pairs = []
        for u in range(len(pi[j+1])):
            pi[j+1][u] = min([pi[j][v]+emissionProbability(Y[u],X[j],em)+transitionProbability(Y[v],Y[u],tm) for v in range(len(pi[j]))])
            pairs.append((u,pi[j+1][u]))
        #print(pairs)
        index = max(pairs,key=lambda item:item[1])[0]
        res_y.append(Y[index])
    #print(pi)
    return res_y




# # To produce the labels for Viterbi algo
# # Input: x, y, emission matrix and transition matrix
# # Output: Sequence
# def dicViterbi(x,y,em,tm):

#     # Number of sentences
#     n = len(x)
#     sequence = []

#     # Initialization of pi dictionary
#     pi = {}
#     pi[0] = 1
#     temp = {}
#     for i in range(len(y)):
#         temp[i] = 0
#     for j in range(n):
#         pi[j+1] = temp
#     pi[n+1] = 0


#     # Perform Viterbi Algo to gather the predicted labels
#     for j in range(n):
#         # For each new sentence start a fresh pairs dic
#         pairs = {}
#         if j == 0:
#             for u in range(0, len(pi[1])):
#                 # Pick maximum score for each entry in pi matrix
#                 pi[1][u] = pi[1][0]*emissionProbability(y[0],x[j],em)*transitionProbability(y[0],y[0],tm)
#                 # Gather corresponding label and score for the maximum score
#                 pairs[u] = pi[1][u]
        
#         else: 
#             for u in range(0, len(pi[j+1])):
#                 # Pick maximum score for each entry in pi matrix
#                 pi[j+1][u] = max([pi[j][v]*emissionProbability(y[u],x[j],em)*transitionProbability(y[v],y[u],tm) for v in range(len(pi[j]))])
#                 # Gather corresponding label and score for the maximum score
#                 pairs[u] = pi[j+1][u]
            
#     # Conduct backtracking
#     index = max(pairs,key=pairs.get)
#     sequence.append(y[index])

#     # # Perform Viterbi Algo to gather the predicted labels
#     # for j in range(n):
#     #     # For each new sentence start a fresh pairs dic
#     #     pairs = {}
#     #     for u in range(0, len(pi[j+1])):
#     #         # Pick maximum score for each entry in pi matrix
#     #         pi[j+1][u] = max([pi[j][v]*emissionProbability(y[u],x[j],em)*transitionProbability(y[v],y[u],tm) for v in range(len(pi[j]))])
#     #         # Gather corresponding label and score for the maximum score
#     #         pairs[u] = pi[j+1][u]
        
#     #     # Conduct backtracking
#     #     index = max(pairs,key=lambda item:item[1])[0]
#     #     sequence.append(y[index])

    
#     #print(" --- ")
#     #print(sequence)
#     return sequence
# # ---



def convertToOutput(path, overall_seq, x):

    outFile = open(path, "w+")
    i = 0
    j = 0
    for sentence in x:
        i += 1
        j = 0
        for word in sentence:
            j += 1
            outFile.write('%s ' % (word))
            outFile.write('%s\n' % (overall_seq[i-1][j-1]))
        outFile.write('\n')
    outFile.close()



# Execution Script --- 
start_time = time.time()

# RUN ONCE FOR FILE CREATION. THEN COMMENT OUT.
# x,y = inputGenXY('./Data/EN/train')
# transitionParameters()
# emissionParameters()
# unique_words = sorted(list(flatten(y)))
# unique_words.remove('O')
# #Since each sentence starts with a prior state of 'O'
# unique_words = ['O'] + unique_words
# unique_words_pd = pd.DataFrame(unique_words)
# unique_words_pd.to_pickle('unique_words')


# Comment the following for the first run. Then uncomment it for all following runs.
x = inputGenX('./Data/EN/dev.in')
emission_matrix = pd.read_pickle('emission_matrix')
transition_matrix = pd.read_pickle('transition_matrix')
label_count = pd.read_pickle('label_count')
# Store the list of words as a global variable
word_list = (emission_matrix.index)
unique_words = pd.read_pickle('unique_words')
unique_words = sorted(list(flatten(unique_words.values.tolist())))
unique_words.remove('O')
#Since each sentence starts with a prior state of 'O'
unique_words = ['O'] + unique_words
overall_seq = []
# ISSUE: NEED TO FIND SOLUTION FOR STORING NESTED LOOPS.
total_len = len(x)
i = 0
for sentence in x:
    #print(sentence)
    output = logViterbi(sentence,unique_words,emission_matrix,transition_matrix)
    overall_seq.append(output)
    i += 1
    print("{} / {}".format(i,total_len))
    # print(output)
    # print('---')

# print(overall_seq)
convertToOutput('./Data/EN/dev.p3.out', overall_seq, x)


stop_time = time.time()
time_taken = stop_time - start_time
mins = time_taken // 60
seconds = time_taken % 60
print('--- Total Time Taken: {} mins {} secs'.format(mins,seconds))
# ---