# %%
from rnn_data import load_ndfa, load_brackets
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from collections import Counter
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_len = 4):
        self.sequences, (self.index_to_word, self.word_to_index)  = load_ndfa(n=150_000)

        # Pad sequences with window lenght + 1 (so you can always predict 1)
        for i in range(len(self.sequences)):
            pad_len = max((sequence_len+1)-len(self.sequences[i]), 0)
            self.sequences[i] = F.pad(torch.tensor(self.sequences[i]), pad = (pad_len ,0), mode='constant', value=0)
        
        # Now convert all these sequences to sequence len sequences
        # So [14, 52, 4, 4, 1, 3], can become multiple sequences
        self.X, self.Y = [], []
        for seq in self.sequences:
            for i in range(len(seq) - sequence_len):
                self.X.append(torch.tensor(seq[i:i+sequence_len]))
                self.Y.append(torch.tensor(seq[i+1:i+sequence_len+1]))

        # Sequence len is the window size that either shortens or pads the given sequences
        self.sequence_len = sequence_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index], self.Y[index]
        )

# %%
class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_width = 128
        self.embeddings = 64
        self.num_layers = 3

        n_vocab = len(dataset.index_to_word)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embeddings,
        )
        self.lstm = nn.LSTM(
            input_size=self.embeddings,
            hidden_size=self.lstm_width,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_width, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def initialize(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_width),
                torch.zeros(self.num_layers, sequence_length, self.lstm_width))


def train(dataset, model, epochs = 10, sequence_len =4 ,batch_size = 32, lr=0.001):
    model.train()

    tb = SummaryWriter("runs/imdb_sequence_0")


    # Load the data and loss function
    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        state_h, state_c = model.initialize(sequence_len)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
            pred=  np.argmax(y_pred.transpose(1, 2)[0].detach().numpy(), axis=0)
            true_y = y[0].detach().numpy()

            accs = [1 if list(a) == list(b) else 0 for a,b in zip(list(pred), list(true_y))]

            if list(pred) != list(true_y):
                print('\ny_pred:',pred, '\ny    :', true_y)

            # tb.add_scalar("Loss/train", loss.item(), batch)
            # tb.add_scalar("Accuracy/val", epoch_val_acc, batch)
            # tb.flush()



# %%
dataset = Dataset()

# %%

model = Model(dataset)

train(dataset, model)
# print(predict(dataset, model, text='Knock knock. Whos there?'))