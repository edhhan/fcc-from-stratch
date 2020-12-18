import numpy as np
import matplotlib.pyplot as plt
import time


# Import data
from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

# Process X data

train_X = train_X.astype('float32').reshape(60000,784).T
test_X = test_X.astype('float32').reshape(10000,784).T
train_X = train_X / 255
test_X = test_X / 255

# Process Y data
train_y_one_hot = np.zeros((train_Y.shape[0], len(np.unique(train_Y))))
train_y_one_hot[np.arange(train_Y.shape[0]), train_Y] = 1 
train_y_one_hot = train_y_one_hot.T

test_y_one_hot = np.zeros((test_Y.shape[0], len(np.unique(test_Y))))
test_y_one_hot[np.arange(test_Y.shape[0]), test_Y] = 1 
test_y_one_hot = test_y_one_hot.T

# Output function
def softmax(x):

    p = np.exp( x - max(x) )

    return p/p.sum(0)


# Home model
class home_model():

    # Constructor
    def __init__(self, L, M, D, K, lr, minibatch_size):

        # Architecture
        self.L = L  # Number of hidden layers
        self.M = M  # Size of hidden layers
        self.D = D  # Input size
        self.K = K  # Output size

        # Training parameters
        self.lr = lr
        self.minibatch_size = minibatch_size

        # Creation of weigth and biais (Structure is array of numpy arrays)
        self.Theta = []

        # Input layer
        W1 = np.random.normal(0, 1, (M, D)) * np.sqrt(2./float(D))  # Weigth of size MxD
        b1 = np.zeros((M, 1))                                       # Biases of size Mx1
        self.Theta.append(np.concatenate((W1,b1),axis=1))           # append concatenation to Theta List

        # Hidden layers
        for l in range(1,L):
            
            Wl = np.random.normal(0, 1, (M, M)) * np.sqrt(2/float(M))  # Weigth of size MxD
            bl = np.zeros((M, 1))
            self.Theta.append(np.concatenate((Wl,bl),axis=1))

        # Output layer
        WL = np.random.normal(0, 1, (K, M)) * np.sqrt(2/float(M))  # Weigth of size KxM
        bL = np.zeros((K, 1))
        self.Theta.append(np.concatenate((WL,bL),axis=1))


        # Training memory 
        self.h_mem = np.zeros((self.M,self.L))
        self.g_prime = np.zeros(self.h_mem.shape)
        self.grad_mem = []
        self.clear_grads()

        self.best_precision = 0
        self.best_model = self.Theta

    # Forward pass
    def propagate_forward(self, x, Theta):

        
        # Compute preactivation
        a = Theta[0].dot(np.append(x,[1]))
        
        # Activation with ReLU
        a[a<0] = 0
        
        # Saving activation value
        self.h_mem[:,0] = a

        # Saving derivative of activation function at preactivation value
        a[a!=0] = 1
        self.g_prime[:,0] = a

        # Propagation in hidden layers
        for l in range(1,self.L):
            
            # Compute preactivation
            a = Theta[l].dot(np.append(self.h_mem[:,l-1],[1]))
            
            # Activation with ReLU
            a[a<0] = 0
            
            # Saving activation values
            self.h_mem[:,l] = a

            # Saving derivative of activation function at preactivation value
            a[a!=0] = 1
            self.g_prime[:,l] = a
        
        # Propagation in output layer
        a_out = Theta[self.L].dot( np.append( self.h_mem[:,self.L-1], [1] ) )

        return softmax(a_out)


    # Training function
    def train_model(self, train_X, train_y_one_hot, nb_epoch, test_X, test_y_one_hot):

        train_losses = []
        test_losses = []
        accuracies = []

        for epoch in range(nb_epoch):

            print(epoch)

            # Shuffling training data
            shuffle = np.random.permutation( train_X.shape[1] )

            train_X = train_X[:, shuffle]
            train_y_one_hot = train_y_one_hot[:, shuffle]

            self.clear_grads()

            for i in range(0, train_X.shape[1]):

                y_pred = self.propagate_forward(train_X[:,i], self.Theta)

                self.update_grads(train_X[:,i], train_y_one_hot[:,i], y_pred)

                # Update condition
                if ( i!=0 and (i % self.minibatch_size == 0 or i == train_X.shape[1]-1) ):
                    
                    self.update_theta()
                    self.clear_grads()
            
            # Compute validation loss for current matrix
            train_loss = 0
            for i in range(0, train_X.shape[1]):
                
                # Basic loss
                y_pred = self.propagate_forward( train_X[:,i], self.Theta)
                train_loss += self.get_loss(train_y_one_hot[:,i], y_pred)

            train_losses.append(train_loss / train_X.shape[1])

            # Compute validation loss for current matrix
            test_loss = 0
            for i in range(0, test_X.shape[1]):
                # Basic loss
                y_pred = self.propagate_forward( test_X[:,i], self.Theta)
                test_loss += self.get_loss(test_y_one_hot[:,i], y_pred)

            test_losses.append(test_loss / test_X.shape[1])

            # Compute accuracy with validation set
            precision = self.get_accuracy(test_X, test_y_one_hot, self.Theta)
            accuracies.append(precision)

            if precision > self.best_precision:
                self.best_model = self.Theta

        return train_losses, test_losses, accuracies

    # Update gradient                 
    def update_grads(self, X, y, y_pred):

        grad = (y - y_pred)

        for l in range(self.L, 0, -1):

            self.grad_mem[l] -= self.get_grads(grad, self.h_mem[:,l-1])

            grad = ((self.Theta[l][:,0:-1]).T).dot(grad) * self.g_prime[:,l-1]

        self.grad_mem[0] -= self.get_grads(grad, X)

    # Reset gradient
    def clear_grads(self):

        self.grad_mem = []
        for i in range(len(self.Theta)):
            self.grad_mem.append(np.zeros(self.Theta[i].shape))

    # Loss function : negative log
    def get_loss(self, y, y_pred):
    
        return -np.dot( y, np.log( y_pred ) )

    # Accuracy function
    def get_accuracy(self, X, y_one_hot, theta):
        
        good_guess = 0
        bad_guess = 0

        for i in range(X.shape[1]):

            y_pred = self.propagate_forward(X[:,i], theta)
            pred = np.argmax(y_pred)

            if((y_one_hot[:,i])[pred]==1):
                good_guess += 1
            else:
                bad_guess += 1
        
        self.h_mem = np.zeros((self.M,self.L))
        self.g_prime = np.zeros(self.h_mem.shape)
        self.grad_mem = []
        self.clear_grads()

        return good_guess / (bad_guess + good_guess) * 100  

    # Getter for grad
    def get_grads(self, grad, h_prev):

        return np.outer(grad, np.append(h_prev, [1]))

    # Update
    def update_theta(self):

        for l in range(self.L+1):
            self.Theta[l] -= self.lr  * self.grad_mem[l] / self.minibatch_size



############
### Main ###
############


# Parameters
nb_epoch = 50
nL = 5         # Number of hidden layers
nM = 300       # Number of nodes per hidden layer

# Clock
t = time.time()

# Creation of model
hmodel = home_model(nL, nM, train_X.shape[0], train_y_one_hot.shape[0], 0.001, 100)

train_losses, test_losses, accuracies = hmodel.train_model(train_X, train_y_one_hot, nb_epoch, test_X, test_y_one_hot)

# Figures
plt.figure()
plt.plot(range(nb_epoch), train_losses, '--b', label='Training')
plt.plot(range(nb_epoch), test_losses, '--r', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Negatif log of likelihood function')
plt.legend()

plt.figure()
plt.plot(range(nb_epoch), accuracies)
plt.ylabel('Precision (%)')
plt.xlabel('Epoch')

elapsed = time.time() - t 

# Output model results
print('Learning time : {}'.format(elapsed))

# Final accuracy with validation set
print(hmodel.get_accuracy(np.append(test_X, axis=1), np.append(test_y_one_hot, axis=1), hmodel.best_model))

plt.show()
