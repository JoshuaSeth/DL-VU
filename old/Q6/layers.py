import math
import numpy as np
from errors import LayerError

class Layer:
    def forward(self, X, y_true, verbose=False):
        pass

    def backward(self, verbose=False):
        pass

    def update(self, alpha, verbose=False):
        pass

    def print(self):
        pass

class Dense(Layer):
    '''A fully connected layer of a certain width, connecting all the input neurons with w to k.'''

    def __init__(self, input_width, width=None, weights=None, bias=None) -> None:
        if width==None and weights==None:
            width= input_width
        if width!=None and weights!= None:
            raise LayerError('Cannot have both preset weights and width. Should either be weights or width')
        

        if width:
            self.width = width
            self.W = np.array([np.array([np.random.normal() for i in range(width)]) for j in range(input_width)])
        if weights:
            self.width = len(weights[0])
            self.W = np.array(weights)
        
        if bias==None: self.bias = np.array([0] * self.width)
        elif len(bias) != len(self.W[0]): raise LayerError('Bias must be of same width as the layer width. Check your given bias, your layer width or your given preset weights.')
        else: self.bias = bias

    def forward(self, X, y_true, verbose=False):
        # Check if correct width
        # if X.shape[0] != len(self.W): raise LayerError('The input should be the same as the initially given dimensions')

        self.X = X  
        self.value = np.matmul(self.X, self.W) + self.bias
        
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nIn:', X, "\nOut:", self.value,'\nWeights:\n', self.W, '\nBias:', self.bias)
        
        return self.value
    
    def get_partial_weights(self, grad, X):
        return np.multiply.outer(grad, X).T
    
    def backward(self, gradient_node_after, context, verbose=False):
        # Gradient for the weights
        # NOTE: This is the least vectorized part of the whole network
        # self.Wgradient =  np.mean([self.get_partial_weights(grad, X) for grad, X in zip(gradient_node_after, self.X)], axis=0)

        # print('gradient node after', gradient_node_after.shape, 'X', self.X.shape, 'desired', self.W.shape)
        self.Wgradient = []
        # for X_idx, X in enumerate(self.X):
        
        # for idx, node in enumerate(self.W):
        #     n = []
        #     self.Wgradient.append(n)
        #     for idx2, w in enumerate(node):
        #         n.append(self.X[idx] * gradient_node_after[idx2])
        # self.Wgradient = np.array(self.Wgradient)
        self.Wgradient = np.multiply.outer(gradient_node_after, self.X).T
        # self.Wgradient = np.mean(self.Wgradient, axis=1)
        # self.Wgradient = np.mean(self.Wgradient, axis=0)

        # print('wgradient', self.Wgradient.shape, 'should be', self.W.shape)

        # print('weight shape', self.W.shape, 'gradient shape', t.shape)
        self.Igradient = np.matmul(self.W, gradient_node_after.T).T
        # print(self.Igradient)

        # print('igradient', self.Igradient.shape, 'must be', self.X.shape[1])
        
        # Gradient for the bias
        self.Bgradient = np.mean(gradient_node_after, axis=0)

        # self.gradient = np.array([self.Igradient, self.Wgradient, self.Bgradient])

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nInput Gradient:', self.Igradient, '\nWeights Gradient:\n', np.array(self.Wgradient), '\nBias Gradient:', self.Bgradient)
        
        return self.Igradient # For the next layer this is the relevant gradient

    def update(self, alpha, verbose=False):
        if verbose: 
            print('\n\n',self.__class__.__name__)
            print('Old weights:\n', self.W)
            print('Change:\n', alpha * self.Wgradient)
            # with np.printoptions(threshold=np.inf):
            #     print('\ninput was\n', self.X, self.X.max(), '\ngradient given is\n',self.gradient_node_after, '\noutput was:\n', self.value)
        
        self.W = self.W  + (alpha * self.Wgradient)

        if verbose:
            print('New weights:\n', self.W)
            print('Old bias:\n', self.bias)
            print('Change:\n', alpha * self.Bgradient)
        self.bias = self.bias  + (alpha * self.Bgradient)

        if verbose:
            print('New bias:\n', self.bias)


class Sigmoid(Layer):
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = 1 / (1 + np.exp(-X)) # X/max to reduce overflows, but cannot just do that at sigmoid

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[1:], '\nIn:', X, "\nOut:", self.value)
        return self.value

    def backward(self, gradient_node_after, context, verbose=False):
        self.gradient = self.value * (1-self.value) *gradient_node_after

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[1:], 'Gradient:', self.gradient)
        return self.gradient


class Softmax(Layer):
    def forward(self, X, y_true, verbose=False):
        self.X = X
        temp = self.X - np.max(self.X) # = Stable softmax

        self.value = np.exp(temp) / (np.sum(np.exp(temp)) / self.X.shape[0]) # This latter part must be there, else if the batch size is 2 the output will all of a sudden be twice as small
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[1:], '\nIn:', X, "\nOut:", self.value)
        return self.value
    
    def backward(self, gradient_node_after,context,  verbose=False):
        self.gradient= np.zeros(self.value.shape)

        p = np.array(context['y'])
        y_idx = np.where(p == 1)
        non_y_idx = np.where(p != 1)

        self.gradient[y_idx] = gradient_node_after[y_idx] * np.mean(self.value[y_idx]  * (1-  self.value[y_idx]), axis=0)
        self.gradient[non_y_idx] = gradient_node_after[non_y_idx] * np.mean(self.value[non_y_idx] * - self.value[non_y_idx], axis=0)

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[1:], 'Gradient:', self.gradient)
        return self.gradient
    



class NLL(Layer):
    '''Negative log likelihood (for Loss)'''
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = np.mean(-np.log(self.X[y_true == 1]+0.001))

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[1:], '\nIn:', X, "\nOut:", self.value)
        return self.value
    
    def backward(self, true_y,context,  verbose=False):
        self.gradient = -1/(self.X+0.001)

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[1:], 'Gradient:', self.gradient)
        return self.gradient