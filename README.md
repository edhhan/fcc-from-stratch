# fcc-from-stratch
A learning project where we attempt to build a Fully-Connected neural network from stratch. The purpose here is clearly not to provide a performant model, but rather to learn the inner mecanisms (backprop, forward pass, etc.) of high-level libraries that abstracts those from the user.

# Model
A Fully-Connected neural network with a variable number of hidden layers and variable number of nodes per layer. The model is applied on the well known [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) dataset. 

Our tests are being done with 5 hidden layers, having 300 nodes each. 

# Packages
```
numpy
```
```
keras
```
```
matplotlib.pyplot
```
```
time
```

# Results
Approximately 4h00 were required to train and test the model, on a standard laptop without any use of the GPU (no parallelization has been implemented), and for only 50 epochs : clearly the model is unefficient. 

With untuned hyperparameters we obtain a final precision on the validation set of approximately 88%. 

<img src="https://github.com/edhhan/fcc-from-stratch/blob/main/results/acc_home.png" width="500" height="300">
<img src="https://github.com/edhhan/fcc-from-stratch/blob/main/results/loss_home.png" width="500" height="300">



# Author
Edward H-Hannan
