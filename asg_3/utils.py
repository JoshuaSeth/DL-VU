import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GlobalMaxPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]


class IMDBDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()

        self.x = [torch.tensor(xi) for xi in x]
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train_epoch(model, data_loader, optimizer, loss_func, device=DEVICE):
    # Make sure to calculate gradients
    model.train(True)
    epoch_loss = 0.0

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward & optim
        optimizer.zero_grad()
        loss = loss_func(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Mean loss over batches
    return epoch_loss / len(data_loader)


def get_score(model, data_loader, loss_func, device=DEVICE):
    # Don't need gradients here
    model.train(False)
    loss = 0.0
    correct = []

    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        out = model.forward(x_batch)
        loss += loss_func(out, y_batch).item()

        # Keep track of correct predictions for accuracy
        correct += (out.argmax(dim=1) == y_batch).tolist()

    # Mean loss over batches
    mean_loss = loss / len(data_loader)
    accuracy = sum(correct) / len(correct)

    return mean_loss, accuracy


def collate_fn_padding(samples):
    x, y = list(zip(*samples))
    return pad_sequence(x, batch_first=True, padding_value=0), torch.tensor(y)
