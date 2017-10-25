import numpy as np
import Neuron

'''The layer class holds the parameters that are required by a layer within the network, including: the weight matrices,
   the centroids for an RBF network, as well as the error values and derivatives used in backpropagation. Additionally
   the layer class is used to propagate inputs through the network'''


class layer:
    def __init__(self, weight_size, activation_function, input_layer = False, output_layer = False, in_sigma=None, k_means=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.weight_size = weight_size  # dimensions of weight matrix
        self.k_means = k_means

        # Holds output and input vector for the layer
        self.outputs = np.zeros(weight_size[0])
        self.inputs = None

        # Create neuron for output calculations
        self.neuron = Neuron.neuron(activation_function, sigma=in_sigma)

        # Create matrices to hold weights, deltas, and derivatives
        self.weights = None
        self.delta_values = None
        self.derivatives = None

        if not input_layer:
            self.inputs = np.zeros(weight_size[0])
            self.delta_values = np.zeros(weight_size[0])

        if not output_layer:
            self.weights = np.random.uniform(-0.2, 0.2, size=weight_size)

        if not output_layer and not input_layer:
            self.derivatives=np.zeros(weight_size[0])

    # Calculate output for layer's neurons
    def calculate_output(self):
        if self.input_layer:
            return self.outputs.dot(self.weights)

        # Run inputs through the activation function
        self.outputs = self.neuron.calculate_output(i_inputs=self.inputs, in_Kvectors=self.k_means)
        if self.output_layer:
            return self.outputs
        else:
            # For hidden layers add bias values, and calculate derivatives
            self.outputs = np.append(self.outputs, 1) # add 1 for bias activation
            self.derivatives = self.neuron.calculate_output(i_inputs=self.inputs, i_want_derivative=True)

            return self.outputs.dot(self.weights)

    def set_delta(self, in_delta):
        self.delta_values = in_delta