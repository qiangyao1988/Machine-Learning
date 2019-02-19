import numpy as np
import matplotlib.pyplot as plt     
from sklearn.datasets.samples_generator import make_blobs

##Create data
(X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)

##Define the number of the training data
n=400

##Create training data 
##x0 = np.full(n, 1.0)
input_data = np.vstack([X]) 
target_data = 1.0/(1 + np.exp(-X))
 
##Two termination conditions
loop_max = 10000   #Maximum number of iterations (prevent dead loops)
epsilon = 1e-3     #Absolute minimum error
 
##Initialization weight
np.random.seed(0)
w = np.random.randn(2)
 
alpha = 0.001      #Step size
diff = 0.0           
error = np.zeros(2) 
count = 0          #Cycle index 
finish = 0         #Ending flag 
i = 1              #Start index

##SGD Alogorithm
while count < loop_max:
    count += 1
##The training data set is traversed and the weight is constantly updated 
    for i in range(n):  
        #The training set is substituted, and the error value is calculated 
        diff = np.dot(w, input_data[i]) - target_data[i] 
        #The SGD algorithm is used to update the weight once and 
        #only one set of training data is used 
        w = w - alpha * diff * input_data[i]
        #Termination conditions: the absolute error of the weight vector 
        #calculated twice is sufficiently small  
    if np.linalg.norm(w - error) < epsilon:     #  
        finish = 1
        break
    else:
        error = w
    i += 1
print ('loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1]))
##Plaint 
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('SGD Scatter Plot') 
plt.xlabel('X')  
plt.ylabel('y')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,edgecolor='black', s=20)
plt.show()


