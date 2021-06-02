import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

''' (1) '''
N   = 5000
mu1 = np.array([0, 2])
mu2 = np.array([0, 0])
cov1= np.array([[4.0, 1.8],
                [1.8, 1.0]])
cov2= np.array([[4.0, 1.2],
                [1.2, 1.0]])

data1 = np.random.multivariate_normal(mu1, cov1, N)
data2 = np.random.multivariate_normal(mu2, cov2, N)
data  = np.vstack([data1, data2])

plt.subplot(1, 3, 1)
plt.scatter(data1[:, 0], data1[:, 1], color = "green")
plt.scatter(data2[:, 0], data2[:, 1], color ="yellow")

''' (2) '''
ml_mu1 = np.average(data1, 0)
ml_mu2 = np.average(data2, 0)

ml_cov1= sum([np.dot((data1[i] - ml_mu1).reshape(2, 1), (data1[i] - ml_mu1).reshape(1, 2)) for i in range(N)])/(N-1)
ml_cov2= sum([np.dot((data2[i] - ml_mu2).reshape(2, 1), (data2[i] - ml_mu2).reshape(1, 2)) for i in range(N)])/(N-1)

x, y = sp.symbols(["x", "y"])
var  = np.array([x, y]).reshape(2, 1)
ml_mu1 = ml_mu1.reshape(2, 1)
ml_mu2 = ml_mu2.reshape(2, 1)

quadtc1 = np.dot(np.dot(var.T, np.linalg.inv(ml_cov2)), var)[0][0]
quadtc2 = np.dot(np.dot(var.T, np.linalg.inv(ml_cov1)), var)[0][0]
quadtc  = +(1/2) * (quadtc1 - quadtc2)
linear1 = np.dot(np.dot(ml_mu1.T, np.linalg.inv(ml_cov1)), var)[0][0]
linear2 = np.dot(np.dot(ml_mu2.T, np.linalg.inv(ml_cov2)), var)[0][0]
linear  = linear1 - linear2
const1  = np.dot(np.dot(ml_mu1.T, np.linalg.inv(ml_cov1)), ml_mu1)
const2  = np.dot(np.dot(ml_mu2.T, np.linalg.inv(ml_cov2)), ml_mu2)
const   = -(1/2) * (const1 - const2) + (1/2) * np.log(np.linalg.det(ml_cov2)/np.linalg.det(ml_cov1))
hyperplane = sp.simplify(quadtc + linear + const)
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

''' (3) '''
pe_2in1 = list(np.array(label1)>=N).count(True)/N
pe_1in2 = list(np.array(label2) <N).count(True)/N
pe = (pe_2in1 + pe_1in2)/2
print(pe)

''' (4) '''
naive_v1 = np.sum((data1 - np.vstack([ml_mu1.reshape(1, 2) for i in range(N)]))**2/(N-1), 0)
naive_v2 = np.sum((data1 - np.vstack([ml_mu1.reshape(1, 2) for i in range(N)]))**2/(N-1), 0)
ml_cov1_naive = np.array([[naive_v1[0], 0], [0, naive_v1[1]]])
ml_cov2_naive = np.array([[naive_v2[0], 0], [0, naive_v2[1]]])

quadtc1_naive = np.dot(np.dot(var.T, np.linalg.inv(ml_cov2_naive)), var)[0][0]
quadtc2_naive = np.dot(np.dot(var.T, np.linalg.inv(ml_cov1_naive)), var)[0][0]
quadtc_naive  = +(1/2) * (quadtc1_naive - quadtc2_naive)
linear1_naive = np.dot(np.dot(ml_mu1.T, np.linalg.inv(ml_cov1_naive)), var)[0][0]
linear2_naive = np.dot(np.dot(ml_mu2.T, np.linalg.inv(ml_cov2_naive)), var)[0][0]
linear_naive  = linear1_naive - linear2_naive
const1_naive  = np.dot(np.dot(ml_mu1.T, np.linalg.inv(ml_cov1_naive)), ml_mu1)
const2_naive  = np.dot(np.dot(ml_mu2.T, np.linalg.inv(ml_cov2_naive)), ml_mu2)
const_naive   = -(1/2) * (const1_naive - const2_naive) + (1/2) * np.log(np.linalg.det(ml_cov2_naive)/np.linalg.det(ml_cov1_naive))
hyperplane_naive = sp.simplify(quadtc_naive + linear_naive + const_naive)
##print(hyperplane_naive)

class1_naive = np.zeros([1, 2])
class2_naive = np.zeros([1, 2])
label1_naive = []
label2_naive = []
for i in range(2*N):
    temp = hyperplane_naive.evalf(subs = {x:data[i, 0], y:data[i, 1]})
    if temp > 0:
        class1_naive = np.vstack([class1_naive, data[i]])
        label1_naive.append(i)
    else:
        class2_naive = np.vstack([class2_naive, data[i]])
        label2_naive.append(i)
class1_naive = class1_naive[1:]
class2_naive = class2_naive[1:]

plt.subplot(1, 3, 3)
plt.scatter(class1_naive[:, 0], class1_naive[:, 1], color = "green")
plt.scatter(class2_naive[:, 0], class2_naive[:, 1], color ="yellow")
plt.show()

pe_2in1_naive = list(np.array(label1_naive)>=N).count(True)/N
pe_1in2_naive = list(np.array(label2_naive) <N).count(True)/N
pe_naive = (pe_2in1_naive + pe_1in2_naive)/2
print(pe_naive)
