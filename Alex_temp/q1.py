# %%
from datetime import datetime

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from rnn_data import load_imdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%
class GlobalMaxPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class RNN(nn.Module):
    def __init__(
        self, n_embeddings: int, embedding_size: int, hidden: int, num_classes: int
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Embedding(n_embeddings, embedding_size),
            nn.Linear(embedding_size, hidden),
            nn.ReLU(),
            GlobalMaxPool(1),
            nn.Linear(hidden, num_classes),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.network(x)


class IMDBDataset(Dataset):
    def __init__(self, x, y, padding_value=0) -> None:
        super().__init__()

        x = [torch.tensor(xi) for xi in x]
        self.x = pad_sequence(x, batch_first=True, padding_value=padding_value)

        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train_epoch(model, data_loader, optimizer, loss_func, device=DEVICE):
    epoch_loss = 0.0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        loss = loss_func(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / (len(data_loader))


# %%
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

train_data = IMDBDataset(x_train, y_train)
val_data = IMDBDataset(x_val, y_val)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

# %%
n_embeddings = len(i2w)
embedding_size = 300
hidden = 300

rnn = RNN(n_embeddings, embedding_size, hidden, 2)
rnn = rnn.to(DEVICE)

# %%
num_epochs = 20
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tb_writer = SummaryWriter(f"runs/imdb_sequence_{timestamp}")

train_loss = []

rnn.train()
for epoch in range(num_epochs):
    epoch_loss = train_epoch(rnn, train_loader, optimizer, loss_func)
    train_loss.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    tb_writer.add_scalar("Loss/train", epoch_loss, epoch)

    # TODO: Validatie

# Epoch 1/20, Loss: 0.5259
# Epoch 2/20, Loss: 0.4319
# Epoch 3/20, Loss: 0.4004
# Epoch 4/20, Loss: 0.3756
# Epoch 5/20, Loss: 0.3563
# Epoch 6/20, Loss: 0.3441
# Epoch 7/20, Loss: 0.3361
# Epoch 8/20, Loss: 0.3313
# Epoch 9/20, Loss: 0.3284
# Epoch 10/20, Loss: 0.3264
# Epoch 11/20, Loss: 0.3252
# Epoch 12/20, Loss: 0.3242
# Epoch 13/20, Loss: 0.3236
# Epoch 14/20, Loss: 0.3231
# Epoch 15/20, Loss: 0.3225
# Epoch 16/20, Loss: 0.3220
# Epoch 17/20, Loss: 0.3217
# Epoch 18/20, Loss: 0.3214
# Epoch 19/20, Loss: 0.3214
# Epoch 20/20, Loss: 0.3212
