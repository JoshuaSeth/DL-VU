import math
from errors import LayerError
import random 

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
            self.W = [[random.gauss(0, 1) for i in range(width)] for j in range(input_width)]
        if weights:
            self.width = len(weights[0])
            self.W = weights
        
        if bias==None: self.bias = [0] * self.width
        elif len(bias) != len(self.W[0]): raise LayerError('Bias must be of same width as the layer width. Check your given bias, your layer width or your given preset weights.')
        else: self.bias = bias

    def forward(self, X, y_true, verbose=False):
        # Check if correct width
        if len(X) != len(self.W): raise LayerError('The input should be the same as the initially given dimensions')

        self.X = X
        self.value = [0] * self.width
        for l, i in enumerate(X):
            for k, j in enumerate(self.W[l]):
                self.value[k] += i * j + self.bias[k]

        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nIn:', X, "\nOut:", self.value,
        '\nWeights:', self.W, 'Bias:', self.bias)
        return self.value
    
    def backward(self, gradient_node_after, context, verbose=False):
        # Gradient for the weights
        self.Wgradient = []
        for idx, node in enumerate(self.W):
            n = []
            self.Wgradient.append(n)
            for idx2, w in enumerate(node):
                n.append(self.X[idx] * gradient_node_after[idx2])

        # Gradient for the inputs
        self.Igradient = []
        for node in self.W:
            n = 0
            for idx2, w in enumerate(node):
               n+=(w * gradient_node_after[idx2])
            self.Igradient.append(n)
        
        # Gradient for the bias
        self.Bgradient = gradient_node_after

        self.gradient = [self.Igradient, self.Wgradient, self.Bgradient]
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nInput Gradient:', self.Igradient, '\nWeights Gradient:', self.Wgradient, '\nBias Gradient:', self.Bgradient)
        return self.Igradient # For the next layer this is the relevant gradient

    def update(self, alpha, verbose=False):
        if verbose: print('\n\n',self.__class__.__name__)
        for idx, node in enumerate(self.W):
            for idx2, w in enumerate(node):
                if verbose: print('Updated weight', idx, idx2, 'from', self.W[idx][idx2], 'to',self.W[idx][idx2] - self.Wgradient[idx][idx2] * alpha, 'by', self.Wgradient[idx][idx2] * alpha)
                self.W[idx][idx2] -= alpha * self.Wgradient[idx][idx2]
        
        for idx, bias in enumerate(self.bias):
            if verbose: print('Updated bias', idx, 'from', self.bias[idx], 'to',self.bias[idx] - self.Bgradient[idx] * alpha, 'by', self.Bgradient[idx] * alpha)
            self.bias[idx] -= alpha * self.Bgradient[idx]


class Sigmoid(Layer):
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = [ 1 / (1 + 2.718281828459045**-i) for i in X]
        self.width = len(X)
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nIn:', X, "\nOut:", self.value)
        return self.value

    def backward(self, gradient_node_after, context, verbose=False):
        self.gradient = [gradient_node_after[idx]* i * (1-i) for idx, i in enumerate(self.value)]
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, 'Gradient:', self.gradient)
        return self.gradient


class Softmax(Layer):
    def forward(self, X, y_true, verbose=False):
        exponents = [math.exp(x) for x in X]
        exp_sum = 0
        for exp in exponents: exp_sum+= exp
        self.value = [x/exp_sum for x in exponents]
        self.width = len(X)
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nIn:', X, "\nOut:", self.value)
        return self.value
    
    def backward(self, gradient_node_after,context,  verbose=False):
        self.gradient= []
        for idx, x in enumerate(self.value):
            if idx == context['y'].index(1):
                self.gradient.append(x * (1-x) * gradient_node_after[idx])
            else: self.gradient.append(-x*x * gradient_node_after[idx])
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, 'Gradient:', self.gradient)
        return self.gradient
    
    

class NLL(Layer):
    '''Negative log likelihood (for Loss)'''
    def forward(self, X, y_true, verbose=False):
        self.X = X
        self.value = -math.log(self.X[y_true.index(1)]) 
        self.width = len(X)
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, '\nIn:', X, "\nOut:", self.value)
        return self.value
    
    def backward(self, true_y,context,  verbose=False):
        self.gradient = [-1/x for x in self.X]
        if verbose: print('\n',self.__class__.__name__, 'Width:', self.width, 'Gradient:', self.gradient)
        return self.gradient