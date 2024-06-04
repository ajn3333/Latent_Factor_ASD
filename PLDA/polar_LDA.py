# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 03:56:43 2023

@author: ANAS AL-NAJI
"""

import numpy as np
import math
from multiprocessing import Pool
import pickle
import os
from scipy import stats


##########################################
#1.Reading Data and preparing the data structure. The discretized_Z variable 
#is the connectivity measure after standardized and discretized. However, in 
#this example, it is only being read from the file that I have attached to you
#in the email
## Importing data


# Define the directory containing the .txt files
directory = '/fs/dss/work/ripo3384/MIND_CORR/MIND_only/'

# Get a list of all .txt files in the directory
txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

# Initialize an empty list to store vectorized lower diagonals
lower_diagonals_vectorized = []

# Loop through each .txt file
for txt_file in txt_files:
    # Read the correlation matrix from the .txt file
    correlation_matrix = np.loadtxt(os.path.join(directory, txt_file))

    # Extract the lower diagonal of the correlation matrix and flatten it
    lower_diagonal_flat = correlation_matrix[np.tril_indices_from(correlation_matrix, k=-1)]

    # Append the flattened lower diagonal to the list
    lower_diagonals_vectorized.append(lower_diagonal_flat)

# Stack all vectorized lower diagonals into a single matrix
lower_diagonals_matrix = np.vstack(lower_diagonals_vectorized)

# This step checks if any of the loaded files contains a nan value and deletes the participant that contains those nans
nan_rows = np.any(np.isnan(lower_diagonals_matrix), axis=1)
lower_diagonals_matrix = lower_diagonals_matrix[~nan_rows]

# Print the shape of the resulting matrix
print("Shape of lower diagonals matrix:", lower_diagonals_matrix.shape)

## Z-Normalizing the data
mean_corrArr_reg = np.mean(lower_diagonals_matrix,0)
std_corrArr_reg = stats.tstd(lower_diagonals_matrix,axis=0) #Used stats.tstd because it seems to be more accurate than np.std
Z = lower_diagonals_matrix - mean_corrArr_reg
Z = Z / std_corrArr_reg

## Discritizing the data
discretized_Z = np.floor(Z*10)


##########################################
#2.Helper functions. In this part I have defied a few helper functions that 
#we will use later in the algorithm

#ensure_not_nan replaces any nan value that gets entered with the second argument
#this is because in some cases the algorithm might run into an issue of deviding by 0
#which results in nan that gets carried out to the next iteration.
def ensure_not_nan(value, default=0.000001):
    if np.isnan(value):
        return default
    return value

#trigamma evaluates the trigamma mathmatical function at x
def trigamma(x):
    x = x+6
    p = 1/(x*x)
    p = (((((0.075757575757576 * p - 0.033333333333333) * p + 0.0238095238095238) * p - 0.033333333333333) * p + 0.166666666666667) * p + 1) / x + 0.5 * p
    for i in range(6):
        x=x-1
        p=1/(x*x)+p
    return p

#digamma evaluates the digamma mathmatical function at x
def digamma(x):
    x = x+6
    p = 1 / (x * x)
    p = (((0.004166666666667 * p - 0.003968253986254) * p + 0.008333333333333) * p - 0.083333333333333) * p
    p=p+np.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6)
    return p

#log_gamma evaluates the natural log of the gamma function at x
def log_gamma(x):
    z = 1/(x*x)
    x = x+6
    z = (((-0.000595238095238 * z + 0.000793650793651) * z - 0.002777777777778) * z + 0.083333333333333) / x
    z = (x - 0.5) * np.log(x) - x + 0.918938533204673 + z - np.log(x - 1) - np.log(x - 2) - np.log(x - 3) - np.log(x - 4) - np.log(x - 5) - np.log(x - 6)
    return z

#Given log(a) and log(b), return log(a + b)
def log_sum(log_a,log_b):
    if log_a < log_b:
        v = log_b+np.log(1 + np.exp(log_a-log_b))
    else:
        v = log_a + np.log(1 + np.exp(log_b - log_a))
    return v

#alhood Calculates the objective function
def alhood(a, alpha_ss, d, k):
    return ((d * (math.lgamma(k * a) - k * math.lgamma(a)) + (a - 1) * alpha_ss))

#d_alhood Calculates the first derivative of the objective function
def d_alhood(a, alpha_ss, d, k):
    return ((d * (k * digamma(k * a) - k * digamma(a)) + alpha_ss))

#d2_alhood Calculates the second derivative of the objective function
def d2_alhood(a, d, k):
    return ((d * (k * k * trigamma(k * a) - k * trigamma(a))))

#rand_init_ss randomly initializes the class_word_pos, class_word, class_total
#variables. These variables are used to calculate the parameters of the model.
#see the rest of the code
def rand_init_ss(class_word_pos,class_word,class_total,num_topics,length_words):
    for k in range(num_topics):
        for n in range(length_words):
            x = 0.5 * (1/length_words + np.random.random())
            y = 0.5 * (1/length_words + np.random.random())
            class_word_pos[k,n] = x
            class_word[k,n] = x + y
            class_total[k] = class_total[k] + class_word[k,n]
    return class_word_pos,class_word,class_total

##########################################
#3.E-Step. In this part I have written the functions that will run one 
#E-step ofthe EM algorithm for one document (One participant) only. Later on 
#I will use a for loop to loop through all the documents.

#polarlda_inference calculates both the gamma and phi parameters that are later
#used to calculate the alpha, beta and rho. See pdf file attached.
#gamma is the document (participant) topic (factor) distribution.
# That is gamma tells us how much a participant loads on each of the factors 
def polarlda_inference(discretized_Z,m,phi,var_gamma,VAR_CONVERGED,VAR_MAX_ITER, alpha, log_prob_w, log_prob_pos):
    converged = 1
    phisum = 0
    likelihood = 0
    likelihood_old = 1
    var_gamma = np.zeros([num_doc,num_topics])
    phi = np.zeros([length_words, num_topics], dtype=np.float32)
    for k in range(num_topics):
        var_gamma[m,k] = alpha + (np.sum(discretized_Z[m])/num_topics)
        digamma_gam[m,k] = digamma(var_gamma[m,k])
        for n in range(length_words):
            phi[n,k] = 1.0/num_topics
    var_iter = 0
    while ((converged > VAR_CONVERGED) & ((var_iter < VAR_MAX_ITER) | (VAR_MAX_ITER == -1))):
        var_iter += 1
        for n in range(length_words):
            phisum = 0
            oldphi = np.zeros(num_topics)
            for k in range(num_topics):
                oldphi[k] = phi[n,k]
                if polarities[m,n]:
                    phi[n, k] = digamma_gam[m,k] + log_prob_w[k,n] + log_prob_pos[k,n]
                else:
                    phi[n, k] = digamma_gam[m,k] + log_prob_w[k,n] + np.log(1 - np.exp(log_prob_pos[k,n]))
                if k > 0:
                    phisum = log_sum(phisum, phi[n,k])
                else:
                    phisum = phi[n,k]
            for k in range(num_topics):
                phi[n, k] = ensure_not_nan(np.exp(phi[n, k] - phisum))
                var_gamma[m,k] += discretized_Z[m,n] * (phi[n,k] - oldphi[k])
                digamma_gam[m,k] = digamma(var_gamma[m,k])
        likelihood = compute_likelihood(discretized_Z, m, phi,var_gamma,alpha, log_prob_w, log_prob_pos)
        converged = (likelihood_old - likelihood) / likelihood_old
        likelihood_old = likelihood
    return likelihood, var_gamma, phi, digamma_gam

#compute_likelihood gives us the contribution of each of the documents to the
#likelihood or the evidence lowerbound. The goal is as with any EM algorithm
#is to maximize this evidence lowerbound.
def compute_likelihood(discretized_Z, m,phi,var_gamma,alpha, log_prob_w,log_prob_pos):
    var_gamma_sum = 0
    dig = np.zeros(num_topics)
    for k in range(num_topics):
        dig[k] = digamma(var_gamma[m,k])
        var_gamma_sum += var_gamma[m,k]
    digsum = digamma(var_gamma_sum)
    likelihood = math.lgamma(alpha * num_topics) - (num_topics * math.lgamma(alpha)) - math.lgamma(var_gamma_sum)
    for k in range(num_topics):
        likelihood += ((alpha - 1) * (dig[k] - digsum)) + math.lgamma(var_gamma[m,k]) - ((var_gamma[m,k] - 1) * (dig[k] - digsum))
        for n in range(length_words):
            if phi[n,k] > 0:
                likelihood += discretized_Z[m,n] * phi[n,k] * ((dig[k] - digsum) - np.log(phi[n,k]) + log_prob_w[k,n])
                if polarities[m,n]:
                    likelihood += discretized_Z[m,n] * phi[n,k] * log_prob_pos[k,n]
                else:
                    likelihood += discretized_Z[m, n] * phi[n, k] * np.log(1 - np.exp(log_prob_pos[k, n]))
    return ensure_not_nan(likelihood)

#doc_e_step runs the E-step for one document and calculates class_word_pos, 
#class_word, class_total variables, which are later used to calculate the 
#the alpha, beta and rho.
def doc_e_step(discretized_Z, m, phi, var_gamma, class_word_pos, class_word, class_total, alpha,alpha_ss,VAR_CONVERGED,VAR_MAX_ITER, log_prob_w, log_prob_pos,polarities):
    likelihood, var_gamma, phi, digamma_gam = polarlda_inference(discretized_Z,m,phi,var_gamma,VAR_CONVERGED,VAR_MAX_ITER,alpha,log_prob_w, log_prob_pos)
    var_gamma_sum = 0
    for k in range(num_topics):
        var_gamma_sum += var_gamma[m,k]
        print(var_gamma[m,k])
        alpha_ss += digamma(var_gamma[m,k])
    alpha_ss = alpha_ss - (num_topics*digamma(var_gamma_sum))
    for n in range(length_words):
        for k in range(num_topics):
            x = discretized_Z[m,n] * phi[n,k]
            class_word_pos[k,n] += x * polarities[m,n]
            class_word[k,n] += x
            class_total[k] += x
    return likelihood, class_word_pos, class_word, class_total, alpha_ss, var_gamma

#opt_alpha finds the optimum alpha, using the newton's method for approximating
#the maximum point
def opt_alpha(alpha_ss, d,k,MAX_ALPHA_ITER):
    init_alpha = 100
    iter = 0

    log_a = np.log(init_alpha)
    df = 0
    while True:
        iter += 1
        a = np.exp(log_a)
        if np.isnan(a):
            init_alpha = init_alpha * 10
            a = init_alpha
            log_a= np.log(a)
        print(a)
        print(alpha_ss)
        df = d_alhood(a, alpha_ss, d, k)
        d2f = d2_alhood(a, d, k)
        log_a = log_a - df / (d2f * a + df)
        if ((abs(df) > NEWTON_THRESH) & (iter < MAX_ALPHA_ITER)):
            break
    return np.exp(log_a)

##########################################
#4.M-Step. This parts defines the functions that runs the M-step.

#polarlda_m_step runs one M-step for all the documents. and calculates the beta
# or the (word|topic) distribution and the rho or the (word|polarity) distribution
# and the alpha.
def polarlda_m_step(class_word_pos, class_word,class_total,alpha_ss,MAX_ALPHA_ITER,Estimate_alpha):
    for k in range(num_topics):
        for n in range(length_words):
            if class_word[k,n] > 0:
                log_prob_w[k,n] = np.log(class_word[k,n]) - np.log(class_total[k])
                if class_word_pos[k,n] > 0:
                    log_prob_pos[k,n] = np.log(class_word_pos[k,n]) - np.log(class_word[k,n])
                    if log_prob_pos[k,n] > -0.0000000000000001:
                        log_prob_pos[k,n] = -0.0000000000000001
                else:
                    log_prob_pos[k,n] = -100
            else:
                log_prob_w[k,n] = -100
                log_prob_pos[k,n] = np.log(0.5)
    if Estimate_alpha:
        alpha = opt_alpha(ensure_not_nan(alpha_ss), num_topics, num_doc,MAX_ALPHA_ITER)
        return log_prob_w,log_prob_pos,alpha
    else:
        return log_prob_w,log_prob_pos,alpha_ss

##########################################
#5. Initialization of variables. I initialized the variables the same way 
#tang et al. initialized them.

num_topics = 3
alpha = 1/num_topics
num_doc = discretized_Z.shape[0]
length_words = discretized_Z.shape[1]
log_prob_w = np.zeros([num_topics, length_words], dtype = np.float32)
log_prob_pos = np.zeros([num_topics, length_words], dtype = np.float32)
class_word_pos = np.zeros([num_topics, length_words], dtype = np.float32)
class_word = np.zeros([num_topics, length_words], dtype = np.float32)
class_total = np.zeros(num_topics, dtype = np.float32)
polarities = np.where(discretized_Z >= 0, 1, 0)
discretized_Z = abs(discretized_Z)
phi = np.zeros([length_words, num_doc], dtype=np.float32)
var_gamma = np.zeros([num_doc,num_topics])
digamma_gam = np.zeros([num_doc,num_topics])
NEWTON_THRESH = 1e-5
VAR_CONVERGED = 1e-12
MAX_ALPHA_ITER = 1000
VAR_MAX_ITER = 100
i = 0
converged_em = 1
EM_CONVERGED = 1e-12
EM_MAX_ITER = 100 #Max number of EM iterations.
likelihood0_old = 1
class_word_pos,class_word,class_total = rand_init_ss(class_word_pos,class_word,class_total,num_topics,length_words)
log_prob_w, log_prob_pos, alpha = polarlda_m_step(class_word_pos, class_word, class_total, alpha, MAX_ALPHA_ITER, Estimate_alpha=0)

##########################################
#6. Run the EM algorithm for all the documents. tang et al. ran a 100 iterations
#I did the same and obtained the same results given the input data
#that they have provided. Keep in mind that each EM iteration will
#take around 10 minutes. This code might take a few hours to run if you choose
#to run the algorithm the full 100 iterations.

variables = {
    'alpha': alpha,
    'log_prob_w': log_prob_w,
    'log_prob_pos': log_prob_pos,
    'class_word_pos': class_word_pos,
    'class_word': class_word,
    'class_total': class_total,
    'polarities': polarities,
    'phi': phi,
    'var_gamma': var_gamma,
    'digamma_gam': digamma_gam
}


### This line is important to be able to run the algorithm in parallel:
if __name__ == "__main__":
    ### Here the EM algorithm runs 100 times or when the change in the likelihood 
    ### between two EM iterations is smaller than 1e-12
    while (((converged_em < 0) | (converged_em > EM_CONVERGED) | (i <= 2)) & (i <= EM_MAX_ITER)):
        class_word_pos = np.zeros([num_topics, length_words], dtype=np.float32)
        class_word = np.zeros([num_topics, length_words], dtype=np.float32)
        class_total = np.zeros(num_topics, dtype=np.float32)
        var_gamma = np.zeros([num_doc,num_topics])
        alpha_ss = 0
        likelihood0 = 0
        ### docs is a list of tuples which contains the data necessary to run
        ### the E-step. It was necessary to put the data into this list of tuples
        ### to be able to use the starmap function, which is necessary to run
        ### the E-step in parallel.
        docs = []
        for doc in range(num_doc):
            docs.append((discretized_Z, doc, phi, var_gamma, class_word_pos, class_word, class_total, alpha,alpha_ss,VAR_CONVERGED,VAR_MAX_ITER, log_prob_w, log_prob_pos,polarities))
        i += 1
        print('##################################################')
        print('RUNNING EM ITERATION:', i)
        ### This part runs the E-step in parallel, since it is the most time 
        ### consuming step.
        ### I have chosen to run it on 4 different processes at once, you can
        ### choose to increase or decrease this number according to your PC
        ### alterativly you can leave the Pool function without any arguments
        ### and python will automatically maximize the number of processors 
        ### used.
        with Pool(25) as pool:
            result = pool.starmap(doc_e_step, docs)
            print('doc_e step running')
            pool.close()
            pool.join()
        for res in result:
            likelihood1, chunk_class_word_pos, chunk_class_word, chunk_class_total, chunk_alpha_ss,var_gamma_chunk = res
            likelihood0 += likelihood1
            class_word_pos += chunk_class_word_pos
            class_word += chunk_class_word
            class_total += chunk_class_total
            alpha_ss += chunk_alpha_ss
            var_gamma += var_gamma_chunk
            print(likelihood0)
            print(alpha_ss)
            print(var_gamma)
        print(likelihood0)
        print(alpha_ss)
        log_prob_w,log_prob_pos,alpha = polarlda_m_step(class_word_pos, class_word,class_total,alpha_ss,MAX_ALPHA_ITER, Estimate_alpha=1)
        print('LIKELIHOOD:', likelihood0)
        converged_em = (likelihood0_old - likelihood0) / (likelihood0_old)
        print('LCHANGE:', converged_em)
        likelihood0_old = likelihood0
        variables_with_iteration = {f'{key}_{i}': value for key, value in variables.items()}
        workspace_file = f'workspace_{i}.pkl'  # Use the iteration number in the file name
        with open(workspace_file, 'wb') as f:
            pickle.dump(variables_with_iteration, f)


#while (((converged_em < 0) | (converged_em > EM_CONVERGED) | (i <= 2)) & (i <= EM_MAX_ITER)):
#    i += 1
#    likelihood0 = 0
#    alpha_ss = 0
#    class_word_pos = np.zeros([num_topics, length_words], dtype=float)
#    class_word = np.zeros([num_topics, length_words], dtype=float)
#    class_total = np.zeros(num_topics, dtype=float)
#    for doc in range(discretized_Z.shape[0]):
#        likelihood1, class_word_pos, class_word, class_total, alpha_ss, var_gamma = doc_e_step(discretized_Z, doc, phi, var_gamma, class_word_pos, class_word, class_total, alpha,alpha_ss,VAR_CONVERGED,VAR_MAX_ITER, log_prob_w, log_prob_pos,polarities)
#        likelihood0 += likelihood1
#    log_prob_w,log_prob_pos,alpha = polarlda_m_step(class_word_pos, class_word,class_total,alpha_ss,MAX_ALPHA_ITER, Estimate_alpha=1)
#    print('LIKELIHOOD:', likelihood0)
#    print('Alpha:', alpha)
#    print(likelihood0)
#    converged_em = (likelihood0_old - likelihood0) / (likelihood0_old)
#    print('LCHANGE:', converged_em)
#    likelihood0_old = likelihood0
#    variables_with_iteration = {f'{key}_{i}': value for key, value in variables.items()}
#    workspace_file = f'workspace_{i}.pkl'  # Use the iteration number in the file name
#    with open(workspace_file, 'wb') as f:
#        pickle.dump(variables_with_iteration, f)


#Printing gamma
#print('GAMMA:', var_gamma)
    
### I commented this part out, but this is a code that would run the EM algorithm
### once, if you would like to try. keep in mind that each time the EM algorithm
### runs once, it takes around 10 minutes to run.
# likelihood0 = 0
# class_word_pos = np.zeros([num_topics, length_words], dtype=np.float64)
# class_word = np.zeros([num_topics, length_words], dtype=np.float64)
# class_total = np.zeros(num_topics, dtype=np.float64)
# alpha_ss = 0
# likelihood1 = 0
# doc = 0
# docs = []
# for doc in range(num_doc):
#     docs.append((discretized_Z, doc, phi, var_gamma, class_word_pos, class_word, class_total, alpha,alpha_ss,VAR_CONVERGED,VAR_MAX_ITER, log_prob_w, log_prob_pos,polarities))
# if __name__ == "__main__":
#     with Pool(8) as pool:
#         result = pool.starmap(doc_e_step, docs)
#         pool.close()
#         pool.join()

#Printing gamma
var_gamma = var_gamma/2
print('GAMMA:')
for row in var_gamma:
    print('   '.join(map(str, row)))
        