from tkinter import *
from tkinter import ttk
import urllib3
import re
import numpy as np
from collections import namedtuple
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
        self.master.title('Classification')
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

        # Activation function selection menu
        labelMenu = Label(self, text="Label Index")
        labelMenu.grid(row=5, column=0)

        menuOptions = ["First", "Last"]
        self.label_index = StringVar(self.master)
        self.label_index.set("              ")

        self.y = OptionMenu(self, self.label_index, *menuOptions)
        self.y.grid(row=5, column=1)

        # Button to load data from UCI repository
        loadButton = Button(self, text="Load!", command=self.loadAction)
        loadButton.grid(row=6, column=1)

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
        options = ["incremental", "batch", "stochastic"]
        self.update_method = StringVar(self.master)
        self.update_method.set("            ")

        self.w = OptionMenu(self, self.update_method, *options)
        self.w.grid(row=4, column=3)

        # Problem type
        problemLabel = Label(self, text="Problem Type")
        problemLabel.grid(row=5, column=2)
        options = ["classification", "regression"]
        self.problem = StringVar(self.master)
        self.problem.set("            ")

        self.x = OptionMenu(self, self.problem, *options)
        self.x.grid(row=5, column=3)

        # Check box if the user wants to incorporate momentum in the weight updates
        self.use_momentum = ttk.Checkbutton(self, text="Momentum")
        self.use_momentum.grid(row=6, column=2)

        # Beta value for momentum term in weight update
        beta_label = Label(self, text="Beta (if momentum selected)")
        beta_label.grid(row=7, column=2)

        self.beta = Entry(self)
        self.beta.grid(row=7, column=3)

        # Button to build and start running network
        build = Button(self, text="Build and Run!", command=self.build_net)
        build.grid(row=8, column=3)

    # Using GUI inputs initialize the network structure
    def build_net(self):
        nodes_per_layer = self.nodes.get().split(',')
        layer_structure = [int(self.featureNumber.get())]

        for layer in nodes_per_layer:
            layer_structure.append(int(layer))

        layer_structure.append(len(self.label_dict))
        net = MLP.network(layer_structure, self.actFunc.get(), self.problem.get())

        # Run the network
        self.run(net)

    # Load data from UCI repository and convert it to list of tuples(inputs, solution)
    def loadAction(self):
        url = self.sourceURL.get()

        http = urllib3.PoolManager()
        response = http.request('GET', url)
        data = response.data.decode("utf-8")
        data_lines = re.split('\n', data)

        for i in range(int(self.numInstances.get())):
            data_lines[i] = re.sub("\s+", ",", data_lines[i].strip())
            features_label = re.split('[, \t]', data_lines[i]) #    data_lines[i].split(',')
            #features_label = features_label[0:int(self.featureNumber.get())+1]
            print(features_label)
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

        # data = re.split('[, \n]', data)
        #
        # num_features = int(self.featureNumber.get())
        # features = []
        #
        # for i in range(int(self.numInstances.get()) * (num_features + 1)):
        #     if (i + 1) % (num_features + 1) != 0:
        #         features.append(float(data[i]))
        #     else:
        #         current_label = np.zeros(len(self.label_dict))
        #         current_label[self.label_dict.get(data[i])] = 1
        #
        #         self.data.append(trial_run(features, current_label))
        #         features = []

        np.random.shuffle(self.data)
        print(self.data)

    def saveLabel(self):
        self.label_dict[self.labelEntry.get()] = self.label_number
        self.label_number += 1
        self.labelEntry.delete(0, END)
        print(self.label_dict)

    # Run network with parameters from GUI, print results of test data set
    def run(self, net):
        learning = float(self.learningRate.get())
        training_number = int(len(self.data)*0.66)
        self.training_data = self.data[0:training_number]
        testing_data = self.data[training_number:-1]

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
                print("Beginning iteration %s!" % i)
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

        for instance in testing_data:
            print(str(net.calculate_outputs(instance.inputs)) + ":   ", end="")
            print(instance.solution)

        exit()
        return net

if __name__ == '__main__':
    root = Tk()
    app = build_GA_Menu(root)
    root.mainloop()