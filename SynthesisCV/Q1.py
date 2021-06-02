import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

''' (1) '''
N   = 500
mu1 = np.array([0, 0])
mu2 = np.array([2, 2])
cov = np.array([[1, 0.25],
                [0.25, 1]])

data1 = np.random.multivariate_normal(mu1, cov, N)
data2 = np.random.multivariate_normal(mu2, cov, N)
data  = np.vstack([data1, data2])

plt.subplot(1, 3, 1)
plt.scatter(data1[:, 0], data1[:, 1], color = "green")
plt.scatter(data2[:, 0], data2[:, 1], color ="yellow")

''' (2) '''
ml_mu1 = np.average(data1, 0)
ml_mu2 = np.average(data2, 0)
ml_mu1 = ml_mu1.reshape(2, 1)
ml_mu2 = ml_mu2.reshape(2, 1)
x, y = sp.symbols(["x", "y"])
var  = np.array([x, y]).reshape(2, 1)

linear1 = np.dot(np.dot(ml_mu1.T, np.linalg.inv(cov)), var)[0][0]
linear2 = np.dot(np.dot(ml_mu2.T, np.linalg.inv(cov)), var)[0][0]
linear  = linear1 - linear2
const1  = np.dot(np.dot(ml_mu1.T, np.linalg.inv(cov)), ml_mu1)
const2  = np.dot(np.dot(ml_mu2.T, np.linalg.inv(cov)), ml_mu2)
const   = -(1/2) * (const1 - const2) + (1/2)
hyperplane = sp.simplify(linear + const)
##print(hyperplane)

class1 = np.zeros([1, 2])
class2 = np.zeros([1, 2])
label1 = []
label2 = []
for i in range(2*N):
    temp = hyperplane.evalf(subs = {x:data[i, 0], y:data[i, 1]})
    if temp > 0:
        class1 = np.vstack([class1, data[i]])
        label1.append(i)
    else:
        class2 = np.vstack([class2, data[i]])
        label2.append(i)
        
class1 = class1[1:]
class2 = class2[1:]

plt.subplot(1, 3, 2)
plt.scatter(class1[:, 0], class1[:, 1], color = "green")
plt.scatter(class2[:, 0], class2[:, 1], color ="yellow")
plt.plot(np.linspace(-3, 5, 10), 2-np.linspace(-3, 5, 10), color = "red", linewidth = "5", linestyle = "--")

''' (3) '''
pe_2in1 = list(np.array(label1)>=500).count(True)/N
pe_1in2 = list(np.array(label2) <500).count(True)/N
pe_equal = (pe_2in1 + pe_1in2)/2
print(pe_equal)

''' (4) '''
ml_cov1= sum([np.dot((data1[i] - ml_mu1.reshape(1, 2)).reshape(2, 1), (data1[i] - ml_mu1.reshape(1, 2)).reshape(1, 2)) for i in range(N)])/(N-1)
ml_cov2= sum([np.dot((data2[i] - ml_mu2.reshape(1, 2)).reshape(2, 1), (data2[i] - ml_mu2.reshape(1, 2)).reshape(1, 2)) for i in range(N)])/(N-1)

quadtc1 = np.dot(np.dot(var.T, np.linalg.inv(ml_cov2)), var)[0][0]
quadtc2 = np.dot(np.dot(var.T, np.linalg.inv(ml_cov1)), var)[0][0]
quadtc  = +(1/2) * (quadtc1 - quadtc2)
linear1 = np.dot(np.dot(ml_mu1.T, np.linalg.inv(ml_cov1)), var)[0][0]
linear2 = np.dot(np.dot(ml_mu2.T, np.linalg.inv(ml_cov2)), var)[0][0]
linear  = linear1 - linear2
const1  = np.dot(np.dot(ml_mu1.T, np.linalg.inv(ml_cov1)), ml_mu1)
const2  = np.dot(np.dot(ml_mu2.T, np.linalg.inv(ml_cov2)), ml_mu2)
const   = -(1/2) * (const1 - const2) + (1/2) * np.log(np.linalg.det(ml_cov2)/np.linalg.det(ml_cov1)) - np.log(0.005)
hyperplane_avg = sp.simplify(quadtc + linear + const)
##print(hyperplane)

class1 = np.zeros([1, 2])
class2 = np.zeros([1, 2])
label1 = []
label2 = []
for i in range(2*N):
    temp = hyperplane_avg.evalf(subs = {x:data[i, 0], y:data[i, 1]})
    if temp > 0:
        class1 = np.vstack([class1, data[i]])
        label1.append(i)
    else:
        class2 = np.vstack([class2, data[i]])
        label2.append(i)
        
class1 = class1[1:]
class2 = class2[1:]

plt.subplot(1, 3, 3)
plt.scatter(class1[:, 0], class1[:, 1], color = "green")
plt.scatter(class2[:, 0], class2[:, 1], color ="yellow")
plt.show()

''' (5) '''
pe_2in1 = list(np.array(label1)>=N).count(True)/N * 0.005
pe_1in2 = list(np.array(label2) <N).count(True)/N
pe_notEqual = (pe_2in1 + pe_1in2)/2
print(pe_notEqual)
