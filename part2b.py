import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import genfromtxt

# Importing training data ---
train = pd.read_fwf('./Data/EN/train', header = None, index = False)[0]
t_ls = []
for i in range(0,len(train)):
    t_ls.append(train[i].split(" "))   
# Convert list with split data to dataframe
df = DataFrame(t_ls, columns = ['x','y'])
print(df)



# trainingdata = open('./Data/EN/train').read().split('\n')

# #list of x(words) and y(labels)
# x = []
# y = []
# for i in range(len(trainingdata)): 
#     if trainingdata[i] != '':
#         word = trainingdata[i].split(' ')[0]
#         label = trainingdata[i].split(' ')[1]
#         x.append(word)
#         y.append(label)
#     print(i)
    
# #helper function - returns unique list of elements from d
# def flatten(d):
#   return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}

# #creates dataframe of unique x rows and unique y columns
# df = pd.DataFrame(index = flatten(x), columns = flatten(y)).fillna(0)
# print(df)

# ---


# Part 2 (a)(b) ---
def EmissionParam(file, x, y, k):
    top = 0
    bot = 0
    # Table with count values of unique pairs of x and y
    df_top = df.groupby(['x','y']).size().reset_index().rename(columns={0:'count'})
    # Table with count values of unique y
    df_bot = df.groupby(['y']).size().reset_index().rename(columns={0:'count'})
    # Get top and bottom values
    print(df_top)
    print(df_bot)
    top = ((df_top.loc[(df_top['y'] == y) & (df_top['x'] == x)])['count'])
    bot = int((df_bot.loc[df_bot['y'] == y])['count'])
    # (b) ---
    if top.empty == True:
        top = k
        return (top/bot)
    # ---
    else:
        return (int(top)/bot)

# Test the function above
e_param1 = EmissionParam(df,"Municipal", "B-NP", k = 0.5)
print("(a)", e_param1)
e_param2 = EmissionParam(df,"#UNK#", "B-NP", k = 0.5)
print("(b)", e_param2)
# ---

# Part 2 (a)(b) ---
def argmax_y(file, x, k): 
    maximum = 0   
    final_y = ''
    for i in range(0,len(file)):
        if x == file['x'][i]:
            y = file['y'][i]
            val = EmissionParam(file, x, y, k)
            if val > maximum:
                maximum = val
                final_y = y
    return maximum, final_y

maximum, y = argmax_y(df, "Municipal", k = 0.5)
print("Max prob", maximum , "\n" , "y-value:", y)

def argmax_y_tbl(file, k):
    dic = {}
    for i in range(0,len(file)):
        print(file['x'][i])
        print(dic)
        if not(file['x'][i] in dic.keys()):
            key = file['x'][i]
            maximum, final_y = argmax_y(file, key, k)
            dic[key] = [maximum, final_y]
        else: pass
    return dic

dic = argmax_y_tbl(df, k = 0.5)
print(dic)
