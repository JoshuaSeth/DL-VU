from layers import Dense, Sigmoid, Softmax, NLL
from data import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

(xtrain, ytrain), (xval, yval), num_cls = load_mnist()


# The way I have set up the neural net I must transofrm 1 to [1,0] and 0 to [0,1] like a one-hot
ytrain = [np.insert(np.zeros(9),y, 1) for y in ytrain]
yval = [np.insert(np.zeros(9),y, 1) for y in yval]

l1 = Dense(input_width=xtrain.shape[1], width = 300) #weights=[[1,1,1], [-1,-1,-1]]
l2 = Sigmoid()
l3 = Dense(input_width=l1.width, width=10) # weights=[[1,1], [-1, -1], [-1, -1]]
l4 = Softmax()
ll = NLL()

network = [l1, l2, l3,l4, ll]

log_forward = False
log_backward = False
log_update = False

alpha = 0.004
batch_size = 4

losses = []
losses_conf_int = []
for epoch in range(30):
    losses_per_item = []
    for idx, row in enumerate(xtrain):
        if(log_forward):print("\nFORWARD")
        X = row
        for layer in network:
            X = layer.forward(X,y_true=ytrain[idx], verbose=log_forward)
        losses_per_item.append(layer.value)
        losses_conf_int.append([epoch, layer.value])

        if(log_backward):print("\n\nBACKWARD")
        context = {'y': ytrain[idx]}
        grad=None
        for layer in network[::-1]:
            grad = layer.backward(grad,context, verbose=log_backward)


        if(log_update):print("\nUPDATE")
        for layer in network:
            grad = layer.update(alpha, verbose=log_update)

    print('Loss:', np.mean(losses_per_item))
    losses.append(np.mean(losses_per_item))

s = sns.lineplot(data=pd.DataFrame(losses_conf_int, columns=['Epoch', 'Loss']), x="Epoch", y="Loss")
s.set(title='Training loss over 20 epochs for MNIST data')

plt.show()
