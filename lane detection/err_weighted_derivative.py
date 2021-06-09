import numpy as np
import matplotlib.pyplot as plt

##sigmoid function to calculate the probability
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

##derivativ of the sigmoid
def sigm_derv(x):
    return x + (1-x)

##Draw a line function
def drw (x1,x2):
    ln = plt.plot(x1,x2)

##generating random points
np.random.seed(0)
npts = 100
bias = np.ones(npts)
top_region = np.array([np.random.normal(10,2,npts),np.random.normal(12,2,npts), bias]).T
bottom_region = np.array([np.random.normal(5,2,npts),np.random.normal(6,2,npts), bias]).T
all_pnts = np.vstack((top_region, bottom_region))
print(all_pnts)
w1 = -0.3
w2 = -0.4
b = 4
line_prmtrs = np.matrix([w1, w2, b]).T
x1 = np.array([bottom_region[:,0].min(), top_region[:,1].max()])  #the horizontal dimensions of the line that extends between the two extremes
x2 = -b / w2 + x1 * (-w1 / w2)

outs = np.array(np.zeros((npts,1)))
outs = np.append(outs,  np.ones((npts,1)),axis = 0)
# print(outs)
##Percentage evaluation based on the current weights and bias associated to the
##test line using sigmoid function
for iterations in range(500000):

    prcntg = sigmoid(np.dot(all_pnts, line_prmtrs))
    err = (outs - prcntg)     ##1 column
    sgm_dv = sigm_derv(prcntg)
    wght_adjst = np.multiply(err, sgm_dv)
    line_prmtrs += np.dot(all_pnts.T , wght_adjst)
    w1 = float(line_prmtrs[0])
    w2 = float(line_prmtrs[1])
    b = float(line_prmtrs[2])


# print (prcntg)
# print(err)
# print (wght_adjst)
print(line_prmtrs)
# print(w1)
# print(w2)
# print(b)

x1 = np.array([bottom_region[:,0].min(), top_region[:,1].max()])  #the horizontal dimensions of the line that extends between the two extremes
x2 = -b / w2 + x1 * (-w1 / w2)


_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color = 'r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color = 'b')
drw(x1,x2)

plt.show()
