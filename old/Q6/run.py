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

# Normalize
xtrain = xtrain / np.max(xtrain)

batch_size = 1
alpha = 0.0001




l1 = Dense(input_width=xtrain.shape[1], width=10) #weights=[[1,1,1], [-1,-1,-1]]
l2 = Sigmoid()
l3 = Dense(input_width=l1.width, width=10) # weights=[[1,1], [-1, -1], [-1, -1]]
l4 = Softmax()
ll = NLL()

network = [l1, l2, l3,l4, ll]

log_forward = False
log_backward = False
log_update = False


# x_train_batches = np.split(xtrain, np.arange(batch_size,len(xtrain),batch_size))
# y_train_batches = np.split(ytrain, np.arange(batch_size,len(ytrain),batch_size))

x_train_batches=[]
y_train_batches=[]
for x in range(0,100,batch_size):
    x_train_batches.append(xtrain[x:x+batch_size])
    y_train_batches.append(ytrain[x:x+batch_size])

losses = []
losses_conf_int = []
val_loss_conf_int = []
for epoch in range(30):
    losses_per_item = []
    for idx, batch in enumerate(xtrain[:500]):
        if(log_forward):print("\nFORWARD")
        X = batch
        for layer in network:
            X = layer.forward(X,y_true=ytrain[idx], verbose=log_forward)
        losses_per_item.append(layer.value)
        losses_conf_int.append([epoch, layer.value])


        X = xval[idx]
        for layer in network:
            X = layer.forward(X,y_true=yval[idx], verbose=log_forward)
        val_loss_conf_int.append([epoch, layer.value])

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


s = sns.lineplot(data=pd.DataFrame(val_loss_conf_int, columns=['Epoch', 'Loss']), x="Epoch", y="Loss")
s.set(title='Validation loss over 20 epochs for MNIST data')

plt.show()
