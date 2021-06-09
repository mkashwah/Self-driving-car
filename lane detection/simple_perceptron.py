import numpy as np

def sigmoid(numm):

    return 1/(1+np.exp(-1*numm))

def sgmd_drvtv(x):
    return x + (1-x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs= np.array([[0,1,1,0]]).T
ons_offs = np.array([[]])
np.random.seed(0)

synaptic_weights = 2* np.random.random((3,1)) - 1

# print ("random satrting synaptic weights")
# print(synaptic_weights)

for iterations in range(1):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    #calculate error
    err = training_outputs - outputs
    wght_adjst = err * sgmd_drvtv(outputs)
    synaptic_weights += np.dot(input_layer.T, wght_adjst)
for vals in outputs:
    if vals <= 0.5:
        ons_offs = np.append(ons_offs, "off")
    else:
        ons_offs = np.append(ons_offs, "on")



print (err)
print(wght_adjst)
print("outputs after training")
print((outputs))
print("\nFinal synaptic weights")
print(synaptic_weights)

print("\nFinal i/o")
print(ons_offs.T)


# print("weight adjustment")
# print(wght_adjst)
