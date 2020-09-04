#Convolutional neural network based on thermodynamic model

Similar to the thermodynamic model, this convolutional neural network treat the nearest neighbor parameters as convolutional kernels and update these parameters through backpropagation when fitting to the binding activities.

##How to use it
    1. Prepare one-hot encoded sequences (seq_train.pickle, seq_test.pickle) and corresponding binding activities (score_train.pickle, score_test.pickle), and the trainable parameters (nn.pickle);
    2. run "python3 Thermo_CNN.py".