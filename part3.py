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
# Smallest value possible
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
           #print(data_pair)
            yij = data_pair.split(" ")[-1]
            xij = data_pair.strip(" "+yij)
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


# Populate the Transition Params Table ---
# Input: No input. But need to run df first.
# Output: transParamsTable stored in a pickle
def transParamsTable():
    global y

    labels = flatten(y)
    transition_matrix = pd.DataFrame(index = labels, columns = labels)
    transition_matrix.fillna(0,inplace=True)
    for i in range(len(y)):
        for j in range(len(y[i])-1):
            first_word = y[i][j]
            second_word = y[i][j+1]
            transition_matrix.at[str(first_word),str(second_word)] +=1
    sumdf = transition_matrix.sum(axis=0)
    transitionParamsTable = transition_matrix.divide(sumdf,axis='index')
    transitionParamsTable = np.log(transitionParamsTable + small)
    print("--- Transition Parameters Table populated ---")
    (transitionParamsTable.sort_index()).to_pickle('transitionParamsTable')
    return transitionParamsTable
# ---



# Emission Parameters function ---
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



# Get a specific  emission probability ---
# Input: label and word (where label -> word) and k-value. emission_matrix
# Output: emission probability
def emissionProbability(label, word, emission_matrix, k=0.5):
    if not(word in word_list):
        return (k / (label_count[label] + k))
    elif word in word_list:
        return emission_matrix.at[word,label]
# ---



# Returns predicted lables for each sentence ---
# Inputs: List of x (input of words), List of Y (labels), Emission Matrix, Transition Matrix
# Outputs: Predicted lables for a list of inputs
def logViterbi(X,Y,em,tm):
    n = len(X)
    res_y = []

    # Initialization
    start = [[1]]
    t_square = [ [ 0 for i in range(len(Y)) ] for j in range(n) ]
    pi = start + t_square

    # Perform Viterbi
    for j in range(n):
        for u in range(len(pi[j+1])):
            pi[j+1][u] = max([pi[j][v]+emissionProbability(Y[u],X[j],em)+transitionProbability(Y[v],Y[u],tm) for v in range(len(pi[j]))])

    # Conduct backtracking
    res_y = [Y[np.argmax(pi[n])]]
    for j in range(n-1,-1,-1):
        maxval = float('-inf')
        label = Y[0]
        for v in range(len(pi[j])):
            val = pi[j][v]+transitionProbability(Y[v],res_y[n-j-1],tm)
            if val > maxval:
                maxval = val
                label = Y[v]
        res_y.append(label)
    ret = res_y[::-1]
    ret.pop(0)
    return ret
# ---



# Converts and stores the output generated into appropriate format for evalResults.py ---
# Inputs: Desired path for output file to be stored, predicted labels generated, list of words x
# Outputs: 
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
# ---



# EXECUTION FUNCTIONS TO TRAIN DATA ---
def run_train(path):

    global x
    global y
    global label_count
    global word_list

    print("--- Training Start ---")
    x,y = inputGenXY(path)
    # Generate Transition Params Table
    transParamsTable()
    # Generate Emission Params Table
    emissionParameters()
    # Lables Generation
    unique_words = sorted(list(flatten(y)))
    # unique_words.remove('O')
    # # Since each sentence starts with a prior state of 'O'
    # unique_words = ['O'] + unique_words
    unique_words_pd = pd.DataFrame(unique_words)
    unique_words_pd.to_pickle('unique_words')
    print("--- Training Complete ---")
    return None
# ---


# EXECUTION FUNCTIONS TO TEST DATA ---
def run_test(pathIn, pathOut):

    global x
    global label_count
    global word_list

    print('--- Testing Start ---')
    x = inputGenX(pathIn)
    # Load Trained Parameters
    emission_matrix = pd.read_pickle('emission_matrix')
    transition_matrix = pd.read_pickle('transitionParamsTable')
    label_count = pd.read_pickle('label_count')
    # Store the list of words as a global variable
    word_list = (emission_matrix.index)
    unique_words = pd.read_pickle('unique_words')
    unique_words = sorted(list(flatten(unique_words.values.tolist())))
    #print(unique_words)
    # unique_words.remove('O')
    # # Since each sentence starts with a prior state of 'O'
    # unique_words = ['O'] + unique_words
    overall_seq = []
    total_len = len(x)
    i = 0
    for sentence in x:
        output = logViterbi(sentence,unique_words,emission_matrix,transition_matrix)
        overall_seq.append(output)
        i += 1
        print("{} / {}".format(i,total_len))
        
    convertToOutput(pathOut, overall_seq, x)
    print('--- Testing Complete ---')
# ---





# Execution Script --- 
###
start_time = time.time()
###


# RUN ONCE FOR FILE CREATION. THEN COMMENT OUT.
run_train('./Data/EN/train')


# Comment the following for the first run. Then uncomment it for all following runs.
run_test('./Data/EN/dev.in', './Data/EN/dev.p3.out')


# To Compare to dev.p3.out to gold standard use evaluation script ---
# RUN THIS: python3 evalResult.py dev.out dev.prediction
# ---



###
stop_time = time.time()
time_taken = stop_time - start_time
mins = time_taken // 60
seconds = time_taken % 60
print('--- Total Time Taken: {} mins {} secs'.format(mins,seconds))
###
# ---