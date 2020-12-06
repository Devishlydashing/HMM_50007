import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import genfromtxt

# Importing training data in a table with respective counts ---
def df(path):
    # Import training data ---
    trainingdata = open(path).read().split('\n')

    #list of x(words) and y(labels)
    x = []
    y = []
    for i in range(len(trainingdata)): 
        if trainingdata[i] != '':
            word = trainingdata[i].split(' ')[0]
            label = trainingdata[i].split(' ')[1]
            x.append(word)
            y.append(label)
        
        
    #helper function - returns unique list of elements from d
    def flatten(d):
        return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}

    #creates dataframe of unique x rows and unique y columns
    df = pd.DataFrame(index = flatten(x), columns = flatten(y)).fillna(0)

    # Aggregate the counts
    for w,lbl in zip(x,y):
        df.at[w,lbl] = df.at[w,lbl] + 1
        #print(w,lbl)
        #print(df.at[w,lbl])

    # Sort output in ascending order
    df = df.sort_index(ascending=True)
    return df
# ---



# Emission parameters ---
# Input: Dataframe and k-value
# Output: df Dataframe table with Emission Probs populated in it & Summ Dataframe table with aggregated (count(y)) values.
# Addition: Will also output Emission Params Table and Summ Table into a Pickle
def EmissionParamsTable(df,k):
        
    # Gather denominator values (count(y))
    summ = df.sum()
    summ = summ.sort_index(ascending=True)
    #print(summ)

    # Divide the numerator  (count(x -> y)) with the denominator
    for col in df.columns:
        # count(x -> y) / count(y)+ k
        df[col] = df[col] / (summ[col]+k)
    #print(df)
    df.to_pickle('emissionParamsTable')
    summ.to_pickle('summTable')
    print("--- Emission Params Table Function Ran ---")
    print("--- emissionParamsTable:")
    print(df)
    print("--- summTable:")
    print(summ)
    return df, summ

# Input: x value, y value and k-value
# Output: Emission Probability for the combination of x and y
def EmissionParams(x,y,k):
    emissionParamsTable = pd.read_pickle('emissionParamsTable')
    summTable = pd.read_pickle('summTable')

    # For cases not in the Test Set
    if not(x in df.index):
        out = k / (summTable[y]+k)
    else:
        out = emissionParamsTable.at[x,y]
    return out
# ---



# Produce arg max ---
# Input: X value, k value
# Output: Y value with highest emission probability & the emission probablilty
def argmax(x, k):
    emissionParamsTable = pd.read_pickle('emissionParamsTable')
    summTable = pd.read_pickle('summTable')
    # For the argmax of a x that is not in the training set will return count(y) with lowest count
    if not(x in emissionParamsTable.index):
        argMaxY = (summTable.idxmin())
        em = EmissionParams(x, argMaxY, k=0.5)
        return em, argMaxY
    argMaxY = (df.loc[x].idxmax())
    em = emissionParamsTable.at[x,argMaxY]
    return em, argMaxY 
# ---



# Generate dev.p2.out for a given dev.in ---
# Input: df & path for dev.in
# Output: dev.p2.out
def genDevOut(df, path):
    devIn = open(path).read().split('\n')
    # Generate empty dev.p2.out file
    outPath = path[:-6]
    outPath = outPath + 'dev.p2.out'
    outFile = open(outPath,"w+")
    
    for line in devIn:
        if line != '':
            em, y  = argmax(df, line, k = 0.5)
            outFile.write("%s %s\n" % (line,y))
        else: 
            outFile.write('\n')
    outFile.close()
# ---



# To Compare to dev.p2.out to gold standard use evaluation script ---
# RUN THIS: python evalResult.py dev.out dev.prediction
# ---












# Execution Script ---
# KEEP OPEN
df = df('./Data/EN/train')

# RUN ONCE FOR FILE CREATION. THEN COMMENT OUT.
# out_df, out_summ = EmissionParamsTable(df, k=0.5)


#out = argmax(df, 'zoom', k = 0.5)
#print(out)

#genDevOut(df, './Data/EN/dev.in')
# ---