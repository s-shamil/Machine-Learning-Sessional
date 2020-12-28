# IMPORTS
import math
import random 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from shutil import rmtree


# Data Reading
with open('data.txt','r') as f:
    rows = f.readlines()
rows = [[float(entry) for entry in row.strip().split(' ')] for row in rows]
X = np.array(rows)

# Output directory
output_dir = 'Outputs'
if os.path.exists(output_dir):
    rmtree(output_dir)

try:
    os.mkdir(output_dir)
except:
    pass


########## PCA ##########

# Standardization
st_scaler = StandardScaler()
X = st_scaler.fit_transform(X)

# Eigen vectors
S = np.matmul(X.T,X)/(X.shape[0]-1) #Covariance matrix
eigen_values, eigen_vectors = np.linalg.eig(S)
eigen_values = np.abs(eigen_values)

# Select top two
top_eigen_idx = (-eigen_values).argsort()[:2] #principal 2 components
top_eigen_values = eigen_values[top_eigen_idx]
top_eigen_vectors = eigen_vectors[:,top_eigen_idx]

# Transform input into selected dimension
X_2d = np.matmul(X, top_eigen_vectors)

# Plot the graph
plt.figure(figsize=(8,8))
plt.title('PCA with 2 principal components') 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.xticks() 
plt.yticks() 
plt.scatter(X_2d[:,0],X_2d[:,1],color='blue', marker="o", facecolors='none')
plt.savefig(output_dir + '/PCA_output.png')
plt.close()

########## EM ##########

def normal(x_vec, mu_vec, sig_mat):
    # ensure Dimensions: x_vec,mu_vec=>(D,1), sig_mat=>(D,D)
    x_vec = x_vec.reshape(-1,1)
    mu_vec = mu_vec.reshape(-1,1)
    
    diff_vec = x_vec - mu_vec
    inv_sig_mat = np.linalg.inv(sig_mat)
    
    tmp = np.matmul(diff_vec.T, inv_sig_mat)
    exp_term = (-0.5)*np.matmul(tmp, diff_vec)
    
    D = x_vec.shape[0]
    det_sig_mat = np.abs(np.linalg.det(sig_mat))
    denominator = math.sqrt(math.pow((2*math.pi),D)*det_sig_mat)
    
    normal_prob = (1/denominator)*math.exp(exp_term)
    return normal_prob

def init_params(X, n_distrib):
    np.random.seed(10)
    # Selecting random points as means
    idx = np.random.choice(range(X.shape[0]), size=n_distrib)
    mu = X[idx]
    
    # Covariance matrices Sigma
    sigma = []
    for i in range(n_distrib):
        sig = np.matmul((X-mu[i]).T,(X-mu[i]))/(X.shape[0]-1)
        sigma.append(sig)
    sigma = np.array(sigma)
    
    
    # mixing coefficient w 
    w1 = np.random.rand(n_distrib)
    sum_w1 = np.sum(w1)
    w = w1/sum_w1
    
    return mu, sigma, w

def log_likelihood(X, mu, sigma, w):
    n_sample = X.shape[0]
    n_distrib = mu.shape[0]
    ll = 0.0
    for i in range(n_sample):
        val = 0.0
        for k in range(n_distrib):
            val_ = w[k]*normal(X[i], mu[k], sigma[k])
            val += val_
        ll += np.log(val)
    return ll

# init params log_likelihood store into prev
n_distrib = 3
n_sample = X_2d.shape[0]
n_attrib = X_2d.shape[1]
mu, sigma, w = init_params(X_2d, n_distrib)
prev_ll = log_likelihood(X_2d, mu, sigma, w)
ll_data = [prev_ll]

eps = 1e-9
p = np.zeros((n_sample,n_distrib))
# loop until convergence
loops = 100
# while(True):
for loop in tqdm(range(loops)):
    # E step
    for i in range(n_sample):
        denominator = np.sum([(w[k]*normal(X_2d[i],mu[k],sigma[k])) for k in range(n_distrib)])
        p[i,:] = np.array([(w[k]*normal(X_2d[i],mu[k],sigma[k])/denominator) for k in range(n_distrib)])
    
    # M step
    for k in range(n_distrib):
        sum_pk = np.sum(p[:,k])
        mu[k] = np.matmul(p[:,k].reshape(1,n_sample), X_2d)/sum_pk
        tmp_sig = np.array([(p[i,k]*np.matmul((X_2d[i]-mu[k]).reshape(n_attrib,1),(X_2d[i]-mu[k]).reshape(1,n_attrib))) for i in range(n_sample)])
        sigma[k] = np.sum(tmp_sig, axis=0)
        sigma[k] /= sum_pk
        w[k] = sum_pk/n_sample
        
    ll = log_likelihood(X_2d, mu, sigma, w)
    ll_data.append(ll)
    if(np.abs(ll-prev_ll)<eps): #no improvement this time
        print('Convergence reached.')
        break
    prev_ll = ll

for k in range(n_distrib):
    print('Distribution #{}'.format(k+1))
    print('mu: ', mu[k])
    print('sigma: \n', sigma[k])
    print('w: ', w[k])
    print('\n')

plt.figure(figsize=(8,8))
plt.ylabel('Log likelihood')
plt.xlabel('Iteration')
plt.title('Log likelihood')
plt.plot(ll_data)
plt.savefig(output_dir + '/log_likelihood_plot.png')
plt.close()

ys = np.argmax(p,axis=1)
plt.figure(figsize=(8,8))
plt.title('EM algorithm with 3 Gaussian distributions') 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.xticks() 
plt.yticks()
colors = ['red', 'green', 'blue']
for x, y in zip(X_2d, ys):
    plt.scatter(x[0], x[1], color=colors[y], marker="o", facecolors='none')

for k in range(n_distrib):
    plt.scatter(mu[k][0], mu[k][1], color='k', marker="x", s=200)
plt.savefig(output_dir + '/EM_output.png')
plt.close()