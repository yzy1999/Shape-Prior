import numpy as np

n = 10000 # n input points
x = np.random.normal(0, 1, [n,3]) # mean, standard variation, shape; generate 10000 input data points
#print(x)

m = 1000 # 10,100,1000
sigma = 0.3 # 0.1, 0.2, 0.3

y = np.zeros([n,m])

for k in range(0,m):
    for i in range(0,n):
        y[i,k] = x[i,0]*(k+1)**2 + x[i,1]*(k+1) + x[i,2] + np.random.normal(0, sigma)

np.save('D:/shape_prior_data/x_basic_m1000_sigma03.npy',x)
np.save('D:/shape_prior_data/y_basic_m1000_sigma03.npy',y)

