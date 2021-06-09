import numpy as np
import matplotlib.pyplot as plt

##sigmoid function to calculate the probability
def sigmoid(score):
    return 1 / (1 + np.exp(-score))    #retuns p


##Draw a line function
def drw (x1,x2):
    ln = plt.plot(x1,x2)
    plt.pause(0.000001)
    ln[0].remove()

##calculating error function
def calc_err(line, points, y):
    m = points.shape[0]
    p = sigmoid(points * line)
    cross_entropy = -(np.log(p).T * y + np.log(1-p).T * (1-y))*(1/m)
    return cross_entropy

def grad_descent(line_para, points, y, alpha):
    m = points.shape[0]
    for i in range(10000):
        p = sigmoid(points * line_para)
        gradient = (points.T * (p-y))*(alpha/m)
        line_para = line_para - gradient
        w1 = line_para.item(0)
        w2 = line_para.item(1)
        b = line_para.item(2)
        x1 = np.array([points[:,0].min(), points[:,1].max()])  #the horizontal dimensions of the line that extends between the two extremes
        x2 = -b / w2 + x1 * (-w1 / w2)
        print(calc_err(line_para,points,y))
        drw (x1,x2)

##generating random points
np.random.seed(0)
npts = 500

y = np.array([np.zeros(npts), np.ones(npts)])    ##output array 2 rows and 1 column
y = y.reshape(npts*2, 1)    ##reshapes to vertical rows of 2*npts and 1 column

bias = np.ones(npts)
top_region = np.array([np.random.normal(10,2,npts),np.random.normal(12,2,npts), bias]).T
bottom_region = np.array([np.random.normal(5,2,npts),np.random.normal(6,2,npts), bias]).T
all_pnts = np.vstack((top_region, bottom_region))
# print(all_pnts.shape[0])
line_prmtrs = np.matrix([np.zeros(3)]).T

# x1 = np.array([bottom_region[:,0].min(), top_region[:,1].max()])  #the horizontal dimensions of the line that extends between the two extremes
# x2 = -b / w2 + x1 * (-w1 / w2)


##Percentage evaluation based on the current weights and bias associated to the
##test line using sigmoid function
print("This is the initial error:")
print(calc_err(line_prmtrs, all_pnts, y))
_, ax = plt.subplots(figsize=(6,6))
ax.scatter(top_region[:,0], top_region[:,1], color = 'r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color = 'b')
# drw(x1,x2)
grad_descent(line_prmtrs, all_pnts, y, 0.05)
plt.show()
