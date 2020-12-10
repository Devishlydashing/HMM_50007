import pandas as pd
import numpy as np


#helper function - read test input (only X)
def readX(file):
    f = open(file)
    f_content = f.read()
    X = []
    xi = []
    for data in f_content.split('\n'):
        
        if data == '':
            if (xi != []):
                X.append(xi)
                xi = []
        else:
            xij = data
            xi.append(xij)
    return X

def readData(path):
    
    f = open(path)
    f_content = f.read()
    X = []  #X is a nested list of sentences, and its words
    Y = []  #Y is the corresponding labels, according to the sentence
    xi = []
    yi = []
    
    for data_pair in f_content.split('\n'):
        if data_pair == '':
            if (xi != []):
                X.append(xi)
                Y.append(yi)
                xi = []
                yi = []
        else:
            xij,yij = data_pair.split(" ")
            xi.append(xij)
            yi.append(yij)
    
    return (X,Y)

def flatten(d):
  return {i for b in [[i] if not isinstance(i, list) else flatten(i) for i in d] for i in b}

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list







def transitionParameters(Y):
    labels = flatten(Y)
    transition_matrix = pd.DataFrame(index = labels, columns = labels)
    transition_matrix.fillna(0,inplace=True)
    for i in range(len(Y)):
        for j in range(len(Y[i])-1):
            first_word = Y[i][j]
            second_word = Y[i][j+1]
            transition_matrix.at[str(second_word),str(first_word)] +=1
    for i in labels:
        count = flatten_list(Y).count(i)
        transition_matrix[i] = transition_matrix[i].div(count)
    return (transition_matrix)    
        




def emissionParameters(X,Y):
    unique_words = flatten(X)
    labels = flatten(Y)
    #print(unique_words)
    #print(labels)
    
    emission_matrix = pd.DataFrame(index = unique_words, columns = labels)
    emission_matrix.fillna(0,inplace=True)
    
    for i in range(len(X)):
        xi = X[i]
        yi = Y[i]
        pairs = []
        pairs = list(zip(*[xi,yi]))  #word-label pairs for each sentence
        
        for j in pairs:
            emission_matrix.at[str(j[0]),str(j[1])] +=1     #count of y->x
    
    label_count = emission_matrix.sum(axis=0)  #count of y
    for i in labels:
        emission_matrix[i] = emission_matrix[i].div(label_count[i])   #probability of x|y
        
    return (emission_matrix)




import numpy as np
a,b = readData('./Data/EN/train')
em = emissionParameters(a,b)
tm = transitionParameters(b)
em = np.log(em+0.000001)
tm = np.log(tm+0.000001)




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







def Viterbi(X,Y,em,tm):
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
            pi[j+1][u] = max([pi[j][v]+emissionProbability(Y[u],X[j],em)+transitionProbability(Y[v],Y[u],tm) for v in range(len(pi[j]))])
            pairs.append((u,pi[j+1][u]))
        #print(pairs)
        index = max(pairs,key=lambda item:item[1])[0]
        res_y.append(Y[index])
    #print(pi)
    return res_y




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










#Set first word to O
unique_words =list(flatten(b))
unique_words.remove('O')
unique_words = ['O'] + unique_words

#print(unique_words)
# for sentence in a:
#     print(sentence)
#     #print(Viterbi(sentence,unique_words,em,tm))
import time
start_time = time.time()

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