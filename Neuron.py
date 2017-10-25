import numpy as np


'''The neuron class is used to handle the activation functions of a given layer in the network, mapping the inputs
    of given neuron to its outputs.'''


class neuron:
    def __init__(self, function_type, sigma=None):
        self.function_type = function_type
        self.sigma = sigma

    def calculate_output(self, i_inputs=None, i_want_derivative=False, in_Kvectors=None):
        # Take inputs to a given neuron and run them through the specified activation function to produce the
        # neuron outputs.
        #
        # If 'i_want_derivative' is set to true, the derivative of the activation function with respect
        # to the neuron inputs will be returned.
        #
        # The derivatives are used when calculating the error in the network during backpropagation.

        output = 0

        # Calculate the output of the neuron based on the activation function
        if self.function_type == "linear":
            output = i_inputs

        elif self.function_type == "sigmoid":
            if i_want_derivative:
                logit = 1 / (1 + np.exp(-i_inputs))
                output = logit * (1 - logit)
            else:
                output = 1 / (1 + np.exp(-i_inputs))

        elif self.function_type == "hyperbolic":
            if i_want_derivative:
                output = 1 - np.tanh(i_inputs)**2
            else:
                output = np.tanh(i_inputs)

        elif self.function_type == "gaussian":
            if i_want_derivative:
                # Derivatives not needed for backprop in RBF network so just return zeros
                output = np.zeros(len(i_inputs))
            else:
                # in_Kvectors is a list of of the centroids for the hidden layer
                num_nodes = len(in_Kvectors)
                output = np.zeros(num_nodes)

                # Calculate the output of the gaussian function using the inputs and the centroids
                for i in range(num_nodes):
                    output[i] = np.exp(-((np.linalg.norm(np.subtract(i_inputs, in_Kvectors[i])) ** 2) / (2 * (self.sigma ** 2))))

        elif self.function_type == "softmax":
            sum = np.sum(np.exp(i_inputs))
            output = np.exp(i_inputs) / sum

        return output
