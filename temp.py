
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


import numpy as np

def simpleViterbi(sentence,labels,em,tm):
    
    n = len(sentence)
    #last word of sentence must be a fullstop
    startword = [[1]]
    body = [ [ 0 for i in range(len(labels)) ] for j in range(len(sentence)) ]
    pi = startword+body
    sentence = ['START'] + sentence
    res_y = []

    for i in range(1,len(sentence)):
        for v in range(len(pi[i])):
            max_val = float('-inf')
            for u in range(len(pi[i-1])):
                # if not(sentence[i] in word_list):
                #     v = label_count.idxmin()
                #     v = labels.index(v)
                val = pi[i-1][u] + transitionProbability(labels[u],labels[v],tm) + emissionProbability(labels[v],sentence[i],em)
                if val > max_val:
                    max_val = val
            pi[i][v] = max_val
    #print(pi)
#     #backtracking
    res_y = [labels[np.argmax(pi[n-1])]]
    for j in range(n-2,-1,-1):
        maxval = float('-inf')
        label = labels[0]
        for v in range(len(pi[j])):
            val = pi[j][v]+transitionProbability(labels[v],res_y[n-j-2],tm)
            if val > maxval:
                maxval = val
                label = labels[v]
        res_y.append(label)
    return res_y

