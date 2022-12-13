# %%
from datetime import datetime
from functools import partial

import numpy as np
import torch
from rnn_data import load_brackets
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from utils import DEVICE, collate_fn_padding


# %%
class BracketsDataset(Dataset):
    def __init__(self, n=150_000):
        x_train, (self.i2w, self.w2i) = load_brackets(n=n)

        self._start = self.w2i[".start"]
        self._end = self.w2i[".end"]

        self.X = [torch.tensor([self._start] + x) for x in x_train]
        self.Y = [torch.tensor(x + [self._end]) for x in x_train]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# %%
class AR(nn.Module):
    def __init__(
        self,
        hidden_size,  # 128
        embedding_size,  # 128
        num_layers,  # 3
        n_vocab,  # len(dataset.i2w)
    ):
        super().__init__()

        self.lstm_width = hidden_size
        self.embeddings = embedding_size
        self.num_layers = num_layers

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

    def forward(self, x, states=None):
        embed = self.embedding(x)
        # hidden & cell states default to zeros if not provided
        output, states = self.lstm(embed, states)
        logits = self.fc(output)

        return logits, states


def train(
    dataset: BracketsDataset,
    model: AR,
    epochs=10,
    batch_size=32,
    lr=0.001,
):
    model.train()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = SummaryWriter(f"runs/brackets_{timestamp}")

    # Load the data and loss function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_fn_padding, padding_value=dataset.w2i[".pad"], pad_y=True
        ),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        for i_batch, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            y_pred, *_ = model(x_batch)
            loss = criterion(y_pred.transpose(1, 2), y_batch)

            loss.backward()
            optimizer.step()

            accs = []
            masked_accs = []

            for i, _ in enumerate(y_batch):
                pred = np.argmax(
                    y_pred.transpose(1, 2).cpu()[i].detach().numpy(), axis=0
                )
                true_y = y_batch[i].cpu().detach().numpy()
                if list(pred) == list(true_y):
                    accs.append(1)
                else:
                    accs.append(0)
                pred = pred[np.nonzero(pred)]
                true_y = true_y[np.nonzero(true_y)]
                if list(pred) == list(true_y):
                    masked_accs.append(1)
                else:
                    masked_accs.append(0)

            if i_batch % 1000 == 0:
                print(
                    {
                        "epoch": epoch,
                        "batch": i_batch,
                        "loss": loss.item(),
                        "acc": np.sum(accs) / len(accs),
                        "masked_acc": np.sum(masked_accs) / len(masked_accs),
                    }
                )

            tb.add_scalar(
                "Loss/train", loss.item(), epoch * (len(dataset) / batch_size) + i_batch
            )
            tb.add_scalar(
                "Accuracy/val",
                np.sum(accs) / len(accs),
                epoch * (len(dataset) / batch_size) + i_batch,
            )
            tb.add_scalar(
                "Masked Accuracy/val",
                np.sum(masked_accs) / len(masked_accs),
                epoch * (len(dataset) / batch_size) + i_batch,
            )
            tb.flush()


# %%
dataset = BracketsDataset()

model = AR(128, 128, 3, len(dataset.i2w))
model.to(DEVICE)

train(dataset, model, epochs=10, batch_size=32)

# %%
