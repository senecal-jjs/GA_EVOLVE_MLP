# These are the functions that are shared by both the testing and the training data.
# The main reason for the existence of this file is to allow the training of the
# networks to exist inside their own class without a circular dependancy graph.

import numpy as np
import MLP

def validate_network(net, validation_data, problem_type):
    """Returns the error (RMSE) of the given network.
    """
    output_vals = []
    true_vals = [test.solution for test in validation_data]

    for testInput in validation_data:
        data_in = testInput.inputs
        if problem_type == "regression":
            out_val = net.calculate_outputs(data_in)[0]
        elif problem_type == "classification":
            out_val = net.calculate_outputs(data_in)

        output_vals.append(out_val)

    if problem_type == "regression":
        error = rmse(output_vals, true_vals)
    elif problem_type == "classification":
        error = accuracy(output_vals, true_vals)

    return error

# Method to calculated the RMSE (error) given an array of network outputs, and an array of the true values
def rmse(predicted, true):
    return np.sqrt(((np.array(predicted) - np.array(true)) ** 2).mean())

def accuracy(predicted, true):
    incorrect = 0
    for i in range(len(predicted)):
        predicted_index = np.argmax(predicted[i])
        true_index = np.argmax(true[i])
        if predicted_index != true_index:
            incorrect += 1

    return incorrect/len(predicted)
