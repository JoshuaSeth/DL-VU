# %%
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rnn_data import load_imdb
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from utils import (
    DEVICE,
    GlobalMaxPool,
    IMDBDataset,
    collate_fn_padding,
    get_score,
    train_epoch,
)


# %%
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
        )

    def forward(self, x):
        return self.network(x)


# %%
# Load IMDB data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

# Handle & convert to Torch datasets
train_data = IMDBDataset(x_train, y_train)
val_data = IMDBDataset(x_val, y_val)

# Dataloaders
train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True, collate_fn=collate_fn_padding
)
val_loader = DataLoader(
    val_data, batch_size=128, shuffle=True, collate_fn=collate_fn_padding
)

# %%
# Model parameters
n_embeddings = len(i2w)
embedding_size = 300
hidden = 300

# Create model
rnn = RNN(n_embeddings, embedding_size, hidden, 2)
rnn = rnn.to(DEVICE)

# %%
# Training parameters
num_epochs = 20
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

# Create TensorBoard writer
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tb_writer = SummaryWriter(f"runs/imdb_sequence_{timestamp}")

train_loss = []
val_loss = []
val_acc = []

for epoch in range(num_epochs):
    # Train
    epoch_train_loss = train_epoch(rnn, train_loader, optimizer, loss_func)
    train_loss.append(epoch_train_loss)

    # Validation
    epoch_val_loss, epoch_val_acc = get_score(rnn, val_loader, loss_func)
    val_loss.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}, Val acc: {epoch_val_acc:.2f}"
    )

    # Write data to TensorBoard
    tb_writer.add_scalar("Loss/train", epoch_train_loss, epoch)
    tb_writer.add_scalar("Loss/val", epoch_val_loss, epoch)
    tb_writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
    tb_writer.flush()

# Epoch 1/20, Train loss: 0.4208, Val loss: 0.3458, Val acc: 0.85
# Epoch 2/20, Train loss: 0.2636, Val loss: 0.2918, Val acc: 0.88
# Epoch 3/20, Train loss: 0.1862, Val loss: 0.3188, Val acc: 0.88
# Epoch 4/20, Train loss: 0.1258, Val loss: 0.2944, Val acc: 0.88
# Epoch 5/20, Train loss: 0.0756, Val loss: 0.3201, Val acc: 0.88
# Epoch 6/20, Train loss: 0.0401, Val loss: 0.3255, Val acc: 0.88
# Epoch 7/20, Train loss: 0.0202, Val loss: 0.3518, Val acc: 0.88
# Epoch 8/20, Train loss: 0.0108, Val loss: 0.3594, Val acc: 0.88
# Epoch 9/20, Train loss: 0.0068, Val loss: 0.3750, Val acc: 0.88
# Epoch 10/20, Train loss: 0.0047, Val loss: 0.3912, Val acc: 0.88
# Epoch 11/20, Train loss: 0.0035, Val loss: 0.4035, Val acc: 0.88
# Epoch 12/20, Train loss: 0.0027, Val loss: 0.4207, Val acc: 0.88
# Epoch 13/20, Train loss: 0.0021, Val loss: 0.4160, Val acc: 0.88
# Epoch 14/20, Train loss: 0.0017, Val loss: 0.4289, Val acc: 0.88
# Epoch 15/20, Train loss: 0.0014, Val loss: 0.4334, Val acc: 0.88
# Epoch 16/20, Train loss: 0.0012, Val loss: 0.4491, Val acc: 0.88
# Epoch 17/20, Train loss: 0.0010, Val loss: 0.4461, Val acc: 0.88
# Epoch 18/20, Train loss: 0.0008, Val loss: 0.4531, Val acc: 0.88
# Epoch 19/20, Train loss: 0.0007, Val loss: 0.4897, Val acc: 0.88
# Epoch 20/20, Train loss: 0.0006, Val loss: 0.4680, Val acc: 0.88

# %%
sns.set()
plt.plot(train_loss, label="Training")
plt.plot(val_loss, label="Validation")
plt.legend()
plt.ylabel("CE Loss")
plt.xlabel("Epoch")
plt.title(f"Loss during training, best accuracy = {max(val_acc)*100:.2f}%")
plt.savefig("q2")
