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
df = pd.DataFrame()
# ---

# Import training data ---
def df_train(path):
    
    global x
    global y
    global lengthDataSet
    global lsStates
    global lengthStates
    global df

    trainingdata = open(path).read().split('\n')
    #list of x(words) and y(labels)
    x.append('')
    y.append("START")
    for i in range(len(trainingdata)):
        if trainingdata[i] != '':
            word = trainingdata[i].split(' ')[0]
            label = trainingdata[i].split(' ')[1]
            x.append(word)
            y.append(label)
        elif trainingdata[i] == '':
            x.append('')
            y.append("STOP")
            x.append('')
            y.append("START")
        
    x.append('')
    y.append("STOP")

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
    #print(lsStates)
    lengthStates = len(lsStates)
    temp = pd.DataFrame(lsStates)
    temp.to_pickle('lsStates')

    lengthDataSet = len(x)
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
    lengthDataSet = len(x)

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

    global transParamsTable

    #rows = ['START']
    rows = []
    columns = []
    for label in flatten(y):
        rows.append(label)
        columns.append(label)
    

    transitionParamsTable = pd.DataFrame(index = rows, columns = columns).fillna(0)
    print(transitionParamsTable)
    labels = copy.deepcopy(y)
    #labels.append('STOP')
    #labels.insert(0,'START')

    nextLabel = 0

    for i in range(len(labels)):
        if nextLabel != 'START':
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
    j_label = x[j]
    print("-----------------------")
    print("-----------------------")
    print("--- Computing Score for j-label: {} & u-label: {}. j: {}] ---".format(j_label,u_label, j))

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
        piVal = 1 * transitionParamsTable.at['START', u_label]
        print("--- Max Score: ------------",piVal)
        viterbiScoreTable.iloc[u,j] = piVal
        viterbiStateTable.iloc[u,j] = 'START'

    
    # Everything else
    elif j > 0 and j < (n+1):
        j_1 = x[j-1]
        print("j - 1 label ---",j_1)
        lsPi = []
        for state in range(len(lsStates)):
            em = EmissionParams(j_1, u_label, k = 0.5)
            #print('EM-----------:', em)
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

    print("--- Length of States:",lengthStates)
    print("--- Length of Data Set:",lengthDataSet)

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
# Input: Which word to backtrack from (3 for 'close')
# Output: Series of sentiments
def backtrack(s):

    global sequence

    viterbiScoreTable = pd.read_pickle('viterbiScoreTable')
    viterbiStateTable = pd.read_pickle('viterbiStateTable')

    # NOTE: NEED SOME HELP VISUALISING THE INDEXING FOR THIS PART.
    s = s - 1

    if s < 0 and s > lengthDataSet:
        print("--- Please select an appropriate value to backtrack from ---")

    if s >= 0 and s <= lengthDataSet:
        for i in range(0, (s+1)):
            
            i = s - i
            if s == (lengthDataSet - 1):
            # Need to integrate stopState variable
                sequence.insert(0, stopState)
                print("--- LAST ONE ---")
                print("--- For '{}' maximum scoring label is '{}' with a score of '{}'. ---".format(x[i], stopState, stopScore))
             

            # Gather index with maximum value. Convert the output to type str. Split the output. ->
            # -> Retrieve the 2nd entry of the string which is the index value. Do some string cleaning
            maximumScoreIndex = viterbiScoreTable.iloc[:, [i]].idxmax().to_string().split('   ')[1][1:]
            #print(maximumScoreIndex)
            #print(x[i])
            #print("--- For '{}' maximum scoring label is '{}' with a score of '{}'. ---".format(x[i], maximumScoreIndex, viterbiScoreTable.at[maximumScoreIndex,x[i]].to_string().split('\n')[0]))
            print("--- For '{}' maximum scoring label is '{}' with a score of '{}'. ---".format(x[i], maximumScoreIndex, ' '))
            sequence.insert(0, maximumScoreIndex)

    sequence.insert(0, 'START')
    print('--- Generated Sequence ---')
    print(sequence)
    print('--- Storing Generated sequence as a Pickle ---')
    temp_df = pd.DataFrame(sequence)
    (temp_df).to_pickle('sequence')
    return None


### ISSUE #5S:
### NOTE: THE STATE TABLE SETS THE STATE TO B-ADJP WHENEVER THE SCORE IS 0. CAN CONSIDER USING A SMALL NUMBER INSTEAD OF 0 CALCULATIONS.
### NOTE: SOME INDEXING ISSUES WITH BACKTRACK.
### NOTE: SOMETIMES THE LOGS WILL SHOW 0 VALUE FOR SOME VERY LOW VALUES.
### RESOLVED:
### NOTE: NEED TO SORT OUT RUNNING VITERBI FOR TEST SET. IT WORKS WELL FOR TRAINING SET. ```Can't seem to enter elif j > 0 and j < (n+1):```


# Execution Script ---
# RUN ONCE FOR FILE CREATION. THEN COMMENT OUT.
df, x, y = df_train('./Data/EN/train')
transParamsTable()



# Comment the following for the first run. Then uncomment it for all following runs.
# NOTE: TAKE NOTE OF FILE PATH ENTERED HERE!!!
#df, x, y = df_test('./Data/EN/train')
#preProc()
#parentPi(27)
#backtrack(27)
# seq = pd.read_pickle('sequence')
# l = list(flatten(seq.values.tolist()))
# print(l)
# ---
