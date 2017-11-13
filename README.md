# GA_EVOLVE_MLP

### Datasets used for this project
All datasests are taken from the UCI machine learning database
+ [Abalone Dataset - Classification](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data)
+ [Concrete Slump Dataset - Regression](https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data)
+ [Yacht Hydrodynamics Data Set - Regression](https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data)
+ [Wine - Classification](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
+ [Letter Recognition - Classification](https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data)

### Application Operation
The user is presented with a GUI consisting of three columns. 

1st Column) Select a local data file to load, if a local file is desired.
            + Select type of problem (classification, regression) from second column
            
2nd Column) Enter required fields to load data directly from the UCI repository
            + 1) UCI source URL, a URL like the ones shown with the datasets above.
            + 2) Class labels, add one label at a time, must match labels in dataset exactly
            + 3) How many features? The number of features associated with the instances
            + 4) How many instances? The number of instances in the dataset
            + 5) Problem Type: Drop down menu providing a selection of classification or regression problem
            + 6) Label Index: Is the label the first or last value in a dataset instance
            + 7) Load!: Load the dataset
            
3rd Column) Network and training parameters 
            + 1) Maximum iterations: Maximum number of iterations or generations to run if training does not reach 
               convergence criteria.
            + 2) Hidden Layer Nodes: The number of nodes to use in the hidden layers. Specified as comma separated values.
                    For example: 3, 7 would create a network with 3 nodes in the 1st hidden layer and 7 nodes in the 2nd 
                    hidden layer
            + 3) Activation Function: Drop down menu that provides choice of sigmoid or hyperbolic tangent function
            + 4) Learning Rate: The learning rate used in weight updates during backpropagation training
            + 5) Momentum: If box is checked momentum will be incorporated in the weight updates during backpropagation
            + 6) Beta: The parameter used to set the influence of momentum in the weight updates. 
                     Typically between 0.5 and 1. 
            + 7) Population Size: Population size to use in the evolutionary algorithms
            + 8) Algorithm Selection: Drop down menu to select the desired training method. 
            + 9) Write ouput: Check box if the user wants to write the test output to a file
            + 10) Build and run: Start training and testing


 


