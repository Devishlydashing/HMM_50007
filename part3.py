import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import genfromtxt
import copy
from part2 import EmissionParams

# Global Variables ---
#list of x(words) and y(labels)
x = []
y = []
lengthDataSet = 0
lsStates = []
lengthStates = 0
viterbiScoreTable = pd.DataFrame()
viterbiStateTable = pd.DataFrame()
transParamsTable = pd.DataFrame()
stopScore = 0
stopState = ''
sequence = []
# ---

# Import training data ---
def df_train(path):
    
    global x
    global y
    global lengthDataSet
    global lsStates
    global lengthStates

    trainingdata = open(path).read().split('\n')

    #list of x(words) and y(labels)
    for i in range(len(trainingdata)): 
        if trainingdata[i] != '':
            word = trainingdata[i].split(' ')[0]
            label = trainingdata[i].split(' ')[1]
            x.append(word)
            y.append(label)
        

    #creates dataframe of unique x rows and unique y columns
    df = pd.DataFrame(index = flatten(x), columns = flatten(y)).fillna(0)

    # Aggregate the counts
    for w,lbl in zip(x,y):
        df.at[w,lbl] = df.at[w,lbl] + 1
        #print(w,lbl)
        #print(df.at[w,lbl])

    # Sort output in ascending order
    df = df.sort_index(ascending=True)
    print("--- Data ingested into df ---")
    # Store list of uniqe states (y)
    lsStates = sorted(list(flatten(y)))
    print(lsStates)
    lengthStates = len(lsStates)
    temp = pd.DataFrame(lsStates)
    temp.to_pickle('lsStates')

    lengthDataSet = len(y)

    return df , x , y
# ---



def df_test(path):
    
    global x
    global lengthDataSet
    global lengthStates

    lsStates = pd.read_pickle('lsStates')
    lsStates = sorted(list(flatten(lsStates.values.tolist())))
    lengthStates = len(lsStates)
    #print(lsStates)

    trainingdata = open(path).read().split('\n')

    #list of x(words) and y(labels)
    for i in range(len(trainingdata)): 
        if trainingdata[i] != '':
            word = trainingdata[i].split(' ')[0]
            x.append(word)

    #creates dataframe of unique x rows and unique y columns
    df = pd.DataFrame(index = flatten(x), columns = lsStates).fillna(0)

    # Aggregate the counts
    for w,lbl in zip(x,y):
        df.at[w,lbl] = df.at[w,lbl] + 1

    # Sort output in ascending order
    df = df.sort_index(ascending=True)
    print("--- Data ingested into df ---")
    lengthDataSet = len(y)

    return df , x , y
# ---



# Helper function - returns unique list of elements from d ---
def flatten(d):
    return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}
# ---


# Populate the Transition Params Table ---
# Input: No input. But need to run df first.
# Output: transParamsTable stored in a pickle
def transParamsTable():
    rows = ['START']
    columns = []
    for label in flatten(y):
        rows.append(label)
        columns.append(label)
        
    columns.append('STOP')
    #print(rows)
    #print(columns)

    transitionParamsTable = pd.DataFrame(index = rows, columns = columns).fillna(0)
    labels = copy.deepcopy(y)
    labels.append('STOP')
    labels.insert(0,'START')
    #print(labels)

    nextLabel = 0

    for i in range(len(labels)):
        if nextLabel != 'STOP':
            label = labels[i]
            nextLabel = labels[i+1]
            transitionParamsTable.at[label,nextLabel] = transitionParamsTable.at[label,nextLabel] + 1
            #print(nextLabel)

    summation = transitionParamsTable.sum()
    summation = summation.sort_index(ascending=True)
    #print(summation)

    for col in transitionParamsTable.columns:
        transitionParamsTable[col] = transitionParamsTable[col] / summation[col]
    
    print("--- Transition Parameters Table populated ---")
    print(transitionParamsTable.sort_index())
    (transitionParamsTable.sort_index()).to_pickle('transitionParamsTable')
# ---


# Pre-processing for Pi function. Creation of viterbi Tables
# Input: NONE
# Output: viterbiScoreTable & viterbiStateTable
def preProc():

    
    global viterbiScoreTable 
    global viterbiStateTable 

    lsStates = pd.read_pickle('lsStates')
    lsStates = sorted(list(flatten(lsStates.values.tolist())))

    # Creation of Score Table
    viterbiScoreTable = pd.DataFrame(index = lsStates, columns = x).fillna(0)
    
    # Creation of State Table
    viterbiStateTable = pd.DataFrame(index = lsStates, columns = x).fillna(0)
    
    print("--- Preprocessing Completed ---")


# Find score of specific node ---
# Input: j - column number (int), u - row number(int), n - length of data y
# Output: Changes made to viterbiScoreTable and viterbiStateTable
def pi(j,u,n):
    
    global viterbiScoreTable
    global viterbiStateTable
    global stopScore
    global stopState
    global lengthStates

    lsStates = pd.read_pickle('lsStates')
    lsStates = sorted(list(flatten(lsStates.values.tolist())))

    # To load pre-process transParamsTable
    transitionParamsTable = pd.read_pickle('transitionParamsTable')
    #print(transitionParamsTable)
    u_label = lsStates[u]
    print("-----------------------")
    print("-----------------------")
    print("--- Computing Score for j:{} & u-label:{}] ---".format(j,u_label))
    print("--- Using Trans Params from Pickle ---")

    # STOP
    if j == (n+1):
        lsPi = []
        j_1 = x[j-1]
        for state in range(lengthStates):
            piVal = viterbiScoreTable.iloc[state, j-1] * transitionParamsTable.at[lsStates[state], u_label]
            lsPi.append(piVal)
        # To generate max score
        maxScore = max(lsPi)
        # Since we do not have space accomodated for STOP in our df
        stopScore = maxScore
        # To generate corresponding prevState
        indxState = lsPi.index(maxScore)
        maxState = lsStates[indxState]
        # Since we do not have space accomodated for STOP in our df
        stopState = maxState
        
    # j == 0
    elif j == 0:
        #lsPi = []
        piVal = 1 * transitionParamsTable.at['START', u_label]
        print("--- Max Score: ------------",piVal)
        viterbiScoreTable.iloc[u,j] = piVal
        viterbiStateTable.iloc[u,j] = 'START'


    # Everything else
    elif j > 0 and j < (n+1):
        j_1 = x[j-1]
        lsPi = []
        for state in range(len(lsStates)):
            em = EmissionParams(j_1, u_label, k = 0.5)
            piVal = viterbiScoreTable.iloc[state,j-1] * transitionParamsTable.at[lsStates[state], u_label] * em
            lsPi.append(piVal)
        # To generate max score
        maxScore = max(lsPi)
        # Since we do not have space accomodated for STOP in our df
        viterbiScoreTable.iloc[u,j] = maxScore
        # To generate corresponding prevState
        indxState = lsPi.index(maxScore)
        #print(indxState)
        maxState = lsStates[indxState]
        # Since we do not have space accomodated for STOP in our df
        viterbiStateTable.iloc[u,j] = maxState
        print("--- Max Score: ------------",maxScore)
        print("--- Max State: ------------",maxState)

# ---



# Parent function for score calculation
# Input: End Node (Will always start from 0)
# Output: Modifications made to viterbiScoreTable and viterbiStateTable
def parentPi(end):

    global viterbiScoreTable
    global viterbiStateTable
    global stopScore
    global stopState
    global lengthStates
    print(lengthStates)

    # Preprocessed nec lengths
    for i in range(0,end):
        for j in range(0,lengthStates):
            pi(j = i, u = j, n = lengthDataSet)

    print("-----------------------")
    print("-----------------------")
    print("--- Score Table:")
    print(viterbiScoreTable)
    print("-----------------------")
    print("-----------------------")
    print("--- State Table:")
    print(viterbiStateTable)
    (viterbiScoreTable.sort_index()).to_pickle('viterbiScoreTable')
    (viterbiStateTable.sort_index()).to_pickle('viterbiStateTable')


# Backtracking funciton 
# Input: Which index to backtrack from
# Output: Series of sentiments
def backtrack(s):

    global sequence

    viterbiScoreTable = pd.read_pickle('viterbiScoreTable')
    viterbiStateTable = pd.read_pickle('viterbiStateTable')

    # NOTE: NEED SOME HELP VISUALISING THE INDEXING FOR THIS PART.
    #s = s + 2

    if s < 1:
        print("--- Please select a higher index to backtrack from ---")
    

    if s == lengthDataSet:
            # Need to integrate stopState variable
            None 

    for i in range(0, s + 1):
        i = s - i

        # Gather index with maximum value. Convert the output to type str. Split the output. ->
        # -> Retrieve the 2nd entry of the string which is the index value. Do some string cleaning
        #print(viterbiScoreTable.iloc[:, [i]].idxmax())
        maximumScoreIndex = viterbiScoreTable.iloc[:, [i]].idxmax().to_string().split('   ')[1][1:]
        sequence.insert(0, maximumScoreIndex)

    sequence.insert(0, 'START')
    print('--- Generated Sequence ---')
    print(sequence)
    print('--- Storing Generated sequence as a Pickle ---')
    temp_df = pd.DataFrame(sequence)
    (temp_df).to_pickle('sequence')
    return None



### NOTE: THE STATE TABLE SETS THE STATE TO B-ADJP WHENEVER THE SCORE IS 0. CAN CONSIDER USING A SMALL NUMBER INSTEAD OF 0 CALCULATIONS.



# Execution Script ---
# KEEP OPEN
# NOTE: TAKE NOTE OF FILE PATH ENTERED HERE!!!
df, x, y = df_test('./Data/EN/dev.in')
preProc()


# RUN ONCE FOR FILE CREATION. THEN COMMENT OUT.
# df, x, y = df_train('./Data/EN/train')
# transParamsTable()


# Comment the following for the first run. Then uncomment it for all following runs.
#parentPi(10)
#backtrack(3)
# seq = pd.read_pickle('sequence')
# l = list(flatten(seq.values.tolist()))
# print(l)

# NOTE: NEED TO SORT OUT RUNNING VITERBI FOR TEST SET. IT WORKS WELL FOR TRAINING SET

parentPi(11)
# ---
