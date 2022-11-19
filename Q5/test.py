from layers import Dense, Sigmoid, Softmax, NLL
from data import load_synth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

(xtrain, ytrain), (xval, yval), num_cls = load_synth()

# The way I have set up the neural net I must transofrm 1 to [1,0] and 0 to [0,1] like a one-hot
ytrain = [[1,0] if y == 1 else [0,1] for y in ytrain]
yval = [[1,0] if y == 1 else [0,1] for y in yval]

X = [1, -1]
l1 = Dense(input_width=len(X), width=4) #weights=[[1,1,1], [-1,-1,-1]]
l2 = Sigmoid()
l3 = Dense(input_width=l1.width, width=8) # weights=[[1,1], [-1, -1], [-1, -1]]
l2b = Sigmoid()
l3b = Dense(input_width=8, width=2)
l4 = Softmax()
ll = NLL()

network = [l1, l2, l3,l2b, l3b, l4, ll]

log_forward = False
log_backward = False
log_update = False

alpha = 0.001

losses = []
losses_conf_int = []
for epoch in range(20):
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

# scale = range(1, len(losses)+1)
# plt.plot(scale, losses)
# plt.title('Training loss over 20 epochs for MNIST data')

s = sns.lineplot(data=pd.DataFrame(losses_conf_int, columns=['Epoch', 'Loss']), x="Epoch", y="Loss")
s.set(title='Training loss over 20 epochs for MNIST data')

plt.show()
