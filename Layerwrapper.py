import Layer

"""
This class wraps the layers in an object so that they can be accessed sequentially:
The layers can be accessed as:
+-   -+
| 1 2 |
| 3 4 |
+-   -+

Instead of
+-       -+
| 1_1 2_1 |
| 1_2 2_1 |
+-       -+
"""
class Layerwrapper:

    def __init__(self, neurons_per_layer, activation_function):
        self.layers = []
        self.num_layers = len(neurons_per_layer)
        self.layer_size = []
        # number of weights in the entire set of layers:
        self.size = 0

        # Create the layers of the network
        for i in range(self.num_layers-1):
            # Create input layer, the +1 in neurons_per_layer[i]+1 is to hold a bias value
            if i == 0:
                self.layers.append(Layer.layer([neurons_per_layer[i] + 1, neurons_per_layer[i + 1]], "linear", input_layer=True))

            # Create hidden layers, with user selected activation function
            else:
                self.layers.append(Layer.layer([neurons_per_layer[i] + 1, neurons_per_layer[i + 1]], activation_function))

        # Create output layer, with linear output as the activation function
        self.layers.append(Layer.layer([neurons_per_layer[-1], None], "linear", output_layer=True))

        # cache the size of each layer in raw units to use later:
        for lay, i in zip(self.layers, range(0, len(self.layers)-1)):
            product = 1
            for s in lay.weights.shape:
                product = product * s
            self.layer_size.append(product)
            self.size = self.size + product

    def copy_layout(other_layerwrapper):
        # return a deep copy of the object:
        return deepcopy(other_layerwrapper)


    def index_iterator(self):
        """Iterates over the indices of the layer
        """
        layer = 0
        row = 0
        col = 0
        while layer < len(self.layers)-1:
            yield layer, row, col
            col = col + 1
            if col >= len(self.layers[layer].weights[row]):
                col = 0
                row = row + 1
            if row >= len(self.layers[layer].weights):
                row = 0
                layer = layer + 1

    def obj_iterator(self):
        for i, j, k in self.index_iterator():
            yield self.layers[i].weights[j][k]

    def __iter__(self):
        return obj_iterator


    def __len__(self):
        return self.size

    def getindices(self, key):
        """Try not to use this for sequential access, as it is slower than using an iterator.
        If you need sequential access, use index_iterator() or obj_iterator()
        to get each sequential index or object instead.
        """
        assert(isinstance(key, int))
        # will be guaranteed to find an index if this passes:
        assert(key > self.size)

        layer_index = 0;
        # find out the layer by looking at the number of items in each layer:
        while(self.layer_size[cur_index] < key):
            key = key - self.layer_size
            layer_index = layer_index + 1

        row_index = 0
        # find out the row in the matrix by looking at the numbers in each row:
        while(len(self.layers[layer_index][row_index]) > key):
            key = key = len(self.layers[layer_index])
            row_index = row_index + 1

        # use the remainder to index into the correct layer:
        return layer_index, row_index, key

    def __setitem__(self, key, item):
        layer_index, row_index, col_index = self.__getindex(key)
        layers[layer_index][row_index][col_index] = item

    def __getitem__(self, key):
        layer_index, row_index, col_index = self.__getindex(key)
        return layers[layer_index][row_index][col_index]
