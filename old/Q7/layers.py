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
            self.W = np.array([[np.random.normal() for i in range(width)] for j in range(input_width)])
        if weights:
            self.width = len(weights[0])
            self.W = np.array(weights)
        
        if bias==None: self.bias = np.array([0] * self.width)
        elif len(bias) != len(self.W[0]): raise LayerError('Bias must be of same width as the layer width. Check your given bias, your layer width or your given preset weights.')
        else: self.bias = bias

    def forward(self, X, y_true, verbose=False):
        # Check if correct width
        if len(X) != len(self.W): raise LayerError('The input should be the same as the initially given dimensions')

        self.X = np.array(X)        
        self.value = np.matmul(self.X, self.W)
        
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nIn:', X, "\nOut:", self.value,'\nWeights:\n', self.W, '\nBias:', self.bias)
        
        return self.value
    
    def backward(self, gradient_node_after, context, verbose=False):
        # Gradient for the weights
        self.Wgradient = np.multiply.outer(gradient_node_after, self.X).T

        # Gradient for the inputs
        self.Igradient = np.matmul(self.W, gradient_node_after)
        
        # Gradient for the bias
        self.Bgradient = gradient_node_after

        # self.gradient = np.array([self.Igradient, self.Wgradient, self.Bgradient])

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nInput Gradient:', self.Igradient, '\nWeights Gradient:\n', np.array(self.Wgradient), '\nBias Gradient:', self.Bgradient)
        
        return self.Igradient # For the next layer this is the relevant gradient

    def update(self, alpha, verbose=False):
        if verbose: 
            print('\n\n',self.__class__.__name__)
            print('Old weights:\n', self.W)
            print('Change:\n', alpha * self.Wgradient)
        
        self.W = self.W  - (alpha * self.Wgradient)

        if verbose:
            print('New weights:\n', self.W)
            print('Old bias:\n', self.bias)
            print('Change:\n', alpha * self.Bgradient)
        self.bias = self.bias  - (alpha * self.Bgradient)

        if verbose:
            print('New bias:\n', self.bias)


class Sigmoid(Layer):
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = 1 / (1 + np.exp(-X))

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[0], '\nIn:', X, "\nOut:", self.value)
        return self.value

    def backward(self, gradient_node_after, context, verbose=False):
        self.gradient = self.value * (1-self.value) *gradient_node_after

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[0], 'Gradient:', self.gradient)
        return self.gradient


class Softmax(Layer):
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = np.exp(X)/ np.sum(np.exp(X))
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[0], '\nIn:', X, "\nOut:", self.value)
        return self.value
    
    def backward(self, gradient_node_after,context,  verbose=False):
        self.gradient= np.zeros(self.value.shape)

        p = np.array(context['y'])
        y_idx = np.where(p == 1)
        non_y_idx = np.where(p != 1)

        self.gradient[y_idx] = gradient_node_after[y_idx] * self.value[y_idx]  * (1-  self.value[y_idx])
        self.gradient[non_y_idx] = gradient_node_after[non_y_idx] * self.value[non_y_idx] * - self.value[non_y_idx]

        t = []
        for idx, x in enumerate(self.value):
            if idx == np.where(context['y'] ==1)[0]:
                t.append(x * (1-x) * gradient_node_after[idx])
            else: t.append(-x*x * gradient_node_after[idx])
        

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[0], 'Gradient:', self.gradient)
        return self.gradient
    
    

class NLL(Layer):
    '''Negative log likelihood (for Loss)'''
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = -math.log(self.X[np.where(y_true ==1)[0]]) 

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[0], '\nIn:', X, "\nOut:", self.value)
        return self.value
    
    def backward(self, true_y,context,  verbose=False):
        self.gradient = -1/self.X

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.X.shape[0], 'Gradient:', self.gradient)
        return self.gradient