# %%
from datetime import datetime
from functools import partial
from typing import Optional

import numpy as np
import torch
from rnn_data import load_brackets
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
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
    dataloader: DataLoader,
    model: AR,
    optimizer: optim.Optimizer,
    epochs: int = 10,
    tb_writer: Optional[SummaryWriter] = None,
):
    model.train()

    # Load the data and loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for i_batch, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            y_pred, *_ = model(x_batch)
            loss = criterion(y_pred.transpose(1, 2), y_batch)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            accs: list[bool] = []
            masked_accs: list[bool] = []

            for y_pred_i, y_batch_i in zip(y_pred, y_batch):
                pred = y_pred_i.argmax(axis=-1).cpu().numpy()
                true_y = y_batch_i.cpu().numpy()
                accs.append((pred == true_y).all())

                pred = pred[np.nonzero(pred)]
                true_y = true_y[np.nonzero(true_y)]
                masked_accs.append(pred.tolist() == true_y.tolist())

            if i_batch % 1000 == 0:
                print(
                    [
                        f"{epoch=}",
                        f"{i_batch=}",
                        f"loss={loss.item():.4f}",
                        f"acc={np.mean(accs):.3f}",
                        f"masked_acc={np.mean(masked_accs):.3f}",
                    ]
                )

            if tb_writer:
                tb_writer.add_scalar(
                    "Loss/train",
                    loss.item(),
                    epoch * len(dataloader) + i_batch,
                )
                tb_writer.add_scalar(
                    "Accuracy/val",
                    np.mean(accs),
                    epoch * len(dataloader) + i_batch,
                )
                tb_writer.add_scalar(
                    "Masked Accuracy/val",
                    np.mean(masked_accs),
                    epoch * len(dataloader) + i_batch,
                )
                tb_writer.flush()


# %%
dataset = BracketsDataset()
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=partial(
        collate_fn_padding, padding_value=dataset.w2i[".pad"], pad_y=True
    ),
)

model = AR(128, 128, 3, len(dataset.i2w))
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tb_writer = SummaryWriter(f"runs/brackets_{timestamp}")

train(dataloader, model, optimizer, epochs=50, tb_writer=tb_writer)

# %%
