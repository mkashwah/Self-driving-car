import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(0)
        self.synaptic_weight = 2*np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1/ (1+ np.exp(-x))

    def sigm_derv(self, x):
        return x * (1-x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iterations in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error*self.sigm_derv(output))
            self.synaptic_weight +=adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weight))
        return output


if __name__ == "__main__":

    neuralnet = NeuralNetwork()

    print("Random synaptic weight")
    print(neuralnet.synaptic_weight)

    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs= np.array([[0,1,1,0]]).T

    neuralnet.train(training_inputs, training_outputs, 300000)
    print("synaptic weights after training ")
    print(neuralnet.synaptic_weight)

    A = str(input("input 1: "))
    B = str(input("input 2: "))
    C = str(input("input 3: "))

    print("new situation input data = ", A, B, C)
    print("output data: ")
    print(neuralnet.think(np.array([A, B, C])))
