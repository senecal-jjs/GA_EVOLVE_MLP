from tkinter import *
from tkinter import ttk
import urllib3
import re
import numpy as np
from collections import namedtuple
import xlsxwriter
import csv
import os, errno, getpass # for file writing
import time
import Darwin
import Genetic
import DiffEvolution
import MLP

trial_run = namedtuple('trial_run', ['inputs', 'solution'])


class build_GA_Menu(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.training_data = []
        self.label_dict = {}
        self.label_number = 0
        self.data = []
        self.init_gui()

    def init_gui(self):
        self.master.title('Train Neural Network!')
        self.pack(fill=BOTH, expand=1)

        # Create entry for source URL
        sourceLabel = Label(self, text="UCI source URL")
        sourceLabel.grid(row=0, column=0)

        self.sourceURL = Entry(self)
        self.sourceURL.grid(row=0, column=1)

        # Create button to add a class label to a list
        classButton = Button(self, text="Add Class Label", command=self.saveLabel)
        classButton.grid(row=2, column=0)

        # Entry for name of class label
        self.labelEntry = Entry(self)
        self.labelEntry.grid(row=2, column=1)

        labelName = Label(self, text="Class Label Name")
        labelName.grid(row=1, column=1)

        # Entry for number of features
        self.featureNumber = Entry(self)
        self.featureNumber.grid(row=3, column=1)

        feature = Label(self, text="How many features?")
        feature.grid(row=3, column=0)

        # Entry for number of instances
        self.numInstances = Entry(self)
        self.numInstances.grid(row=4, column=1)

        feature = Label(self, text="How many instances?")
        feature.grid(row=4, column=0)

        # Problem type
        problemLabel = Label(self, text="Problem Type")
        problemLabel.grid(row=5, column=0)
        options = ["classification", "regression"]
        self.problem = StringVar(self.master)
        self.problem.set("            ")

        self.x = OptionMenu(self, self.problem, *options)
        self.x.grid(row=5, column=1)

        # Where is label located in dataset? This will provide a menu to select a location
        labelMenu = Label(self, text="Label Index")
        labelMenu.grid(row=6, column=0)

        menuOptions = ["First", "Last"]
        self.label_index = StringVar(self.master)
        self.label_index.set("              ")

        self.y = OptionMenu(self, self.label_index, *menuOptions)
        self.y.grid(row=6, column=1)

        self.write_output = ttk.Checkbutton(self, text="Write Output")
        self.write_output.grid(row=7, column=0)

        # Button to load data from UCI repository
        loadButton = Button(self, text="Load!", command=self.loadAction)
        loadButton.grid(row=7, column=1)

        # Entry for number of iterations
        iterationsLabel = Label(self, text="Maximum iterations")
        iterationsLabel.grid(row=0, column=2)

        self.iterations = Entry(self)
        self.iterations.grid(row=0, column=3)

        # Number of nodes per layer
        nodesLabel = Label(self, text="Hidden Layer Nodes")
        nodesLabel.grid(row=1, column=2)

        self.nodes = Entry(self)
        self.nodes.grid(row=1, column=3)

        # Activation function selection menu
        menuLabel = Label(self, text="Activation Function")
        menuLabel.grid(row=2, column=2)

        menuOptions = ["sigmoid", "hyperbolic"]
        self.actFunc = StringVar(self.master)
        self.actFunc.set("              ")

        self.w = OptionMenu(self, self.actFunc, *menuOptions)
        self.w.grid(row=2, column=3)

        # Learning rate
        learningLabel = Label(self, text="Learning Rate")
        learningLabel.grid(row=3, column=2)

        self.learningRate = Entry(self)
        self.learningRate.grid(row=3, column=3)

        # Update method
        updateLabel = Label(self, text="Update method")
        updateLabel.grid(row=4, column=2)
        update_options = ["incremental", "batch", "stochastic"]
        self.update_method = StringVar(self.master)
        self.update_method.set("            ")

        self.t = OptionMenu(self, self.update_method, *update_options)
        self.t.grid(row=4, column=3)

        # Check box if the user wants to incorporate momentum in the weight updates
        self.use_momentum = ttk.Checkbutton(self, text="Momentum")
        self.use_momentum.grid(row=5, column=2)

        # Beta value for momentum term in weight update
        beta_label = Label(self, text="Beta (if momentum selected)")
        beta_label.grid(row=6, column=2)

        self.beta = Entry(self)
        self.beta.grid(row=6, column=3)

        # Select which algorithm you want to use
        alg_label = Label(self, text="Algorithm Selection")
        alg_label.grid(row=7, column=2)
        alg_options = ["Backpropagation", "Genetic Algorithm", u"\u03bc" + "+" + u"\u03bb" + " ES", "Differential Evolution"]
        self.alg_selection = StringVar(self.master)
        self.alg_selection.set("            ")

        self.z = OptionMenu(self, self.alg_selection, *alg_options)
        self.z.grid(row=7, column=3)

        # Button to build and start running network
        build = Button(self, text="Build and Run!", command=self.approx_function)
        build.grid(row=8, column=3)

    # Using GUI inputs initialize the network structure
    def get_mlp_layers(self):
        nodes_per_layer = self.nodes.get().split(',')
        layer_structure = [int(self.featureNumber.get())]

        for layer in nodes_per_layer:
            layer_structure.append(int(layer))

        layer_structure.append(len(self.label_dict))
        return layer_structure

    # Load data from UCI repository and convert it to list of tuples(inputs, solution)
    def loadAction(self):
        url = self.sourceURL.get()

        http = urllib3.PoolManager()
        response = http.request('GET', url)
        data = response.data.decode("utf-8")
        data_lines = re.split('\n', data)

        for i in range(int(self.numInstances.get())):
            data_lines[i] = re.sub("\s+", ",", data_lines[i].strip())
            features_label = re.split('[, \t]', data_lines[i])
            features = []
            current_label = np.zeros(len(self.label_dict))

            if self.label_index.get() == "First":
                if self.problem.get() == "classification":
                    current_label[self.label_dict.get(features_label[0])] = 1
                elif self.problem.get() == "regression":
                    current_label = float(features_label[0])

                for j in range(1, len(features_label)):
                    features.append(float(features_label[j]))

            elif self.label_index.get() == "Last":
                if self.problem.get() == "classification":
                    current_label[self.label_dict.get(features_label[-1])] = 1
                elif self.problem.get() == "regression":
                    current_label = float(features_label[-1])

                for j in range(len(features_label)-1):
                    features.append(float(features_label[j]))

            self.data.append(trial_run(features, current_label))

        np.random.shuffle(self.data)
        print(self.data)

    def saveLabel(self):
        self.label_dict[self.labelEntry.get()] = self.label_number
        self.label_number += 1
        self.labelEntry.delete(0, END)
        # print(self.label_dict)

    # ================== METHODS TO RUN AND TEST ALGORITHMS ============================================================

    def approx_function(self):
        # training_cut = int(0.66*len(self.data))
        # validation_cut = int(0.8*len(self.data))
        # self.training_data = self.data[0:training_cut]
        # self.validation_data = self.data[training_cut:validation_cut]
        # self.testing_data = self.data[validation_cut:len(self.data)]
        #
        # if self.alg_selection.get() == "Backpropagation":
        #     self.run_backprop()
        # elif self.alg_selection.get() == "Genetic Algorithm":
        #     self.run_GA()
        self.data = np.array(self.data)
        data_folds = np.array_split(self.data, 10)
        self.print_starting_info() # Still needs to be implemented

        for i in range(10):

            print("Starting fold " + str(i + 1) + " of 10...")
            self.training_data = []
            self.testing_data = []
            self.validation_data = []
            [self.testing_data.append(trial_run(item[0], item[1])) for item in data_folds[i]]
            [self.validation_data.append(trial_run(item[0], item[1])) for item in data_folds[i - 1]]
            for j in range(10):
                if j != i and j != i - 1:
                    [self.training_data.append(trial_run(item[0], item[1])) for item in data_folds[j]]

            if self.alg_selection.get() == "Backpropagation":
                self.run_backprop()

            if self.alg_selection.get() == "Genetic Algorithm":
                self.run_GA()

            if self.alg_selection.get() == u"\u03bc" + "+" + u"\u03bb" + " ES":
                self.run_ES()

            if self.alg_selection.get() == "Differential Evolution":
                self.run_diff()

            print("----------------------------------------")

        exit()

    def run_backprop(self):
        net_layers = self.get_mlp_layers()
        net = MLP.network(net_layers, self.actFunc.get(), self.problem.get())
        net_rmse = self.train_backprop(net)
        self.test_network(net_rmse[0], rmse_vals=net_rmse[1])

    def run_GA(self):
        net_layers = self.get_mlp_layers()
        ga = Genetic.genetic_algorithm.create_instance(50, net_layers, self.actFunc.get(), self.problem.get())
        net_rmse = self.train_GA(ga)
        self.test_network(net_rmse[0], rmse_vals=net_rmse[1])

    def train_GA(self, ga_instance):
        RMSE = []
        best_network = object
        best_rmse = 999

        # For number of specified generations evolve the network population
        for i in range(int(self.iterations.get())):
            if i % 5 == 0:
                # Calculate the rmse of the fittest individual in the population, and append to list of rmse at each
                # generation
                if self.problem.get() == "regression":
                    print("Beginning generation " + str(i) + " of " + self.iterations.get() + "...with rmse of: " + str(best_rmse))
                    if best_rmse < 2:
                        break
                elif self.problem.get() == "classification":
                    print("Beginning generation " + str(i) + " of " + self.iterations.get() + "...percent incorrect: " + str(best_rmse))
                    if best_rmse < 0.05: # 5% incorrect
                        break

                best_rmse = sys.maxsize
                for individual in ga_instance.population:
                    current_net = ga_instance.create_mlp(individual[0:-1])
                    current_rmse = self.validate_network(current_net)

                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_network = current_net

                RMSE.append(best_rmse)

            # GA parameter order: mutation rate, crossover rate, Num individuals for tournament, training data
            ga_instance.evolve(0.3, 0.8, 15, self.training_data)

        return best_network, RMSE

    # Train network using backpropagation
    def train_backprop(self, net):
        learning = float(self.learningRate.get())
        RMSE = []
        error = 999

        # Set momentum to true if momentum was selected in the GUI
        momentum = False
        beta = None
        for state in self.use_momentum.state():
            if state == "selected":
                momentum = True
                beta = float(self.beta.get())
                print("Momentum in use!")

        for i in range(int(self.iterations.get())):
            if i % 100 == 0:
                if self.problem.get() == "regression":
                    print("Beginning generation " + str(i) + " of " + self.iterations.get() + "...with rmse of: " + str(error))
                    if error < 2:
                        break
                elif self.problem.get() == "classification":
                    print("Beginning generation " + str(i) + " of " + self.iterations.get() + "...percent incorrect: " + str(error))
                    if error < 0.05:
                        break

            np.random.shuffle(self.training_data)

            if self.update_method.get() == "incremental":
                net.train_incremental(self.training_data, learning, use_momentum=momentum, beta=beta)

            elif self.update_method.get() == "batch":
                net.train_batch(self.training_data, learning, use_momentum=momentum, beta=beta)

            elif self.update_method.get() == "stochastic":
                batch_size = int(np.sqrt(len(self.testing_data)))
                num_batches = int(int(self.iterations.get()) / batch_size)
                net.train_stochastic(self.training_data, batch_size, num_batches, learning, use_momentum=momentum,
                                     beta=beta)

            error = self.validate_network(net)
            RMSE.append(error)

        return net, RMSE

    # Method to calculated the RMSE (error) on a validation data set during training
    def validate_network(self, net):
        output_vals = []
        true_vals = [test.solution for test in self.validation_data]

        for testInput in self.validation_data:
            data_in = testInput.inputs
            if self.problem.get() == "regression":
                out_val = net.calculate_outputs(data_in)[0]
            elif self.problem.get() == "classification":
                out_val = net.calculate_outputs(data_in)

            output_vals.append(out_val)

        if self.problem.get() == "regression":
            error = self.rmse(output_vals, true_vals)
        elif self.problem.get() == "classification":
            error = self.accuracy(output_vals, true_vals)

        return error

    # Method to test the performance of the network on a test data set, after training has completed.
    def test_network(self, net, rmse_vals=None):
        ''' Given the trained net, calculate the output of the net
            Print the root mean square error to the console by default
            If write output is set, create a CSV with the test inputs,
            outputs, and other statistics '''

        input_vals = []
        output_vals = []
        true_vals = [test.solution for test in self.testing_data]

        for testInput in self.testing_data:
            data_in = testInput.inputs
            input_vals.append(data_in)
            if self.problem.get() == "regression":
                out_val = net.calculate_outputs(data_in)[0]
            elif self.problem.get() == "classification":
                out_val = net.calculate_outputs(data_in)

            output_vals.append(out_val)

        if self.problem.get() == "regression":
            error = self.rmse(output_vals, true_vals)
            print("RMSE: %f\n" % error)
        elif self.problem.get() == "classification":
            percent_accurate = self.accuracy(output_vals, true_vals)
            print("Percent Incorrect: %f\n" % percent_accurate)

        write = False
        for state in self.write_output.state():
            if state == "selected":
                write = True
        if write:
            self.create_csv(input_vals, output_vals, true_vals, rmse_vals);

    # Method to calculated the RMSE (error) given an array of network outputs, and an array of the true values
    def rmse(self, predicted, true):
        return np.sqrt(((np.array(predicted) - np.array(true)) ** 2).mean())

    def accuracy(self, predicted, true):
        incorrect = 0
        for i in range(len(predicted)):
            predicted_index = np.argmax(predicted[i])
            true_index = np.argmax(true[i])
            if predicted_index != true_index:
                incorrect += 1

        return incorrect/len(predicted)

    def create_csv(self, input_vals, output_vals, true_vals, rmse_vals=None):
        ''' Create a csv file with the test inputs, calculated outputs,
                    true values and relevant statistics. '''

        user = getpass.getuser()
        time_start = time.strftime("%m_%d_%H_%M_%S")
        print("Writing output at time: " + time_start)

        folder_dir = os.path.abspath("./outputs")
        # Make output directory
        try:
            os.makedirs(folder_dir)
            print("Output directory created at " + folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file_name = os.path.join(folder_dir, user + "_" + self.alg_selection.get() + "_" + time_start + ".xlsx")
        print("Writing output to " + file_name)

        # wrt = csv.writer((open(file_name), "wb"))
        # wrt.writerow([str(self.label_dict), "Input Values", "Network Output", "True Values", "Validation Error"])

        workbook = xlsxwriter.Workbook(file_name)
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', "# File created at time " + time_start + " by " + user + " using " + self.alg_selection.get())
        worksheet.write('A2', "Label Key")
        worksheet.write('A3', "Dataset URL or File")
        worksheet.write('B3', self.sourceURL.get())
        worksheet.write('B2', str(self.label_dict))
        worksheet.write('A4', "Input Values")
        worksheet.write('B4', "Network Output")
        worksheet.write('C4', "True Values")
        worksheet.write('D4', "Validation Error")

        row = 5
        for i in range(len(input_vals)):
            worksheet.write(row, 0, str(input_vals[i]))
            worksheet.write(row, 1, str(output_vals[i]))
            worksheet.write(row, 2, str(true_vals[i]))
            row += 1

        row = 5
        for i in range(len(rmse_vals)):
            worksheet.write(row, 3, rmse_vals[i])
            row += 1

        workbook.close()

        with open("Parameters.txt", "w") as text_file:
            print("Nodes per hidden layer: {}".format(self.nodes.get()), file=text_file)
            print("Activation Function: {}".format(self.actFunc.get()), file=text_file)
            print("Training Iterations/Generations: {}".format(self.iterations.get()), file=text_file)
            print("Update Method: {}".format(self.update_method.get()), file=text_file)
            print("Learning rate: {}".format(self.learningRate.get()), file=text_file)

        print("Done writing file.")

    # Method to print the parameters of a given test to the console
    def print_starting_info(self):
        if self.alg_selection.get() == "Genetic Algorithm":
            print("Starting training through Genetic Algorithm\n------------------------------------------------")

            # Print out what was just done:
            print("Nodes per hidden layer: %s" % self.nodes.get())
            print("Activation function: %s" % self.actFunc.get())
            print("Training generations: %s\n" % self.iterations.get())

        if self.alg_selection.get() == "Backpropagation":
            print("Starting training through Backpropagation\n------------------------------------------------")
            print("Nodes per hidden layer: %s" % self.nodes.get())
            print("Activation function: %s" % self.actFunc.get())
            print("Update method: %s" % self.update_method.get())
            print("Learning rate: %s" % self.learningRate.get())
            print("Training iterations: %s\n" % self.iterations.get())


if __name__ == '__main__':
    root = Tk()
    app = build_GA_Menu(root)
    root.mainloop()



    # test = Genetic.genetic_algorithm.create_instance(1000, [2, 5, 1], 'sigmoid', 'regression')
    #
    # #Test function x1 + x2
    # trial_run = namedtuple('trial_run', ['inputs', 'solution'])
    # data = []
    # for i in range(500):
    #     x = np.random.uniform(0, 3)
    #     y = np.random.uniform(0, 3)
    #     data.append(trial_run([x,y], x+y))
    #
    # test_data = []
    # for i in range(500):
    #     x = np.random.uniform(0, 3)
    #     y = np.random.uniform(0, 3)
    #     test_data.append(trial_run([x,y], x+y))
    #
    # for individual in test.population:
    #     print(individual)
    #
    # print()
    # print("Evolving")
    #
    # for i in range(25):
    #     if i % 5 == 0:
    #         print("Generation %s!" % i)
    #         fitness = []
    #         for individual in test.population:
    #             fitness.append(test.fitness(individual[0:-1], test_data))
    #         print(np.min(fitness))
    #     #     for i in range(5):
    #     #         print(test.population[np.random.randint(0, 50)])
    #     test.evolve(0.3, 0.5, 5, data)
    #
    # fitness = []
    # for individual in test.population:
    #     fitness.append(test.fitness(individual[0:-1], test_data))
    #
    # print(np.min(fitness))
    #
    # print()
    # print("Backprop Result!")
    # net = MLP.network([2, 5, 1], 'sigmoid', 'regression')
    #
    # for i in range(500):
    #     np.random.shuffle(data)
    #     net.train_incremental(data, 0.001, use_momentum=False, beta=None)
    #
    # summed_error = 0
    # for instance in test_data:
    #     network_output = net.calculate_outputs(instance.inputs)
    #     summed_error += np.sqrt((network_output - instance.solution) ** 2)
    #
    # print(summed_error / len(test_data))
