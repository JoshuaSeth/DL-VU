import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from data_rnn import load_imdb
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from util import GlobalMaxPool, IMDBDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, num_embeddings: int, embedding_size: int, hidden: int, num_classes: int):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size)
        self.lin1 = nn.Linear(embedding_size, hidden)
        self.relu = nn.ReLU()
        self.globalmaxpool = GlobalMaxPool(1)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.lin1(x))
        x = self.globalmaxpool(x)
        x = self.softmax(self.lin2(x))
        return x




def main():
    # argument parser
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--embedding_size", default=300)
    argparser.add_argument("--hidden_size", default=300)
    argparser.add_argument("--lr", default=0.001)
    argparser.add_argument("--epochs", default=5)
    args = argparser.parse_args()

    # Load Data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    train_data = IMDBDataset(x_train, y_train)
    val_data = IMDBDataset(x_val, y_val)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

    model = MLP(
        num_embeddings=len(i2w),
        embedding_size=args.embedding_size,
        hidden=args.hidden_size,
        num_classes=numcls
    )
    num_epochs = 5
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # to log losses
    train_loss = []
    val_loss = []
    val_accuracy = []

    # tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_writer = SummaryWriter(f"runs/imbb_MLP_{timestamp}")

    # training
    for epoch in range(args.epochs):
        model.train(True)
        t_loss = []
        # train model
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()
            t_loss.append(loss.item())

        epoch_train_loss = sum(t_loss) / len(t_loss)
        train_loss.append(epoch_train_loss)

        # validation
        model.train(False)
        v_loss = []
        correct = []

        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)
            v_loss.append(loss_func(y_pred, y).item())
            correct += (y_pred.argmax(dim=1) == y).tolist()

        epoch_validation_loss = sum(v_loss) / len(v_loss)
        val_loss.append(epoch_validation_loss)
        epoch_validation_accuracy = sum(correct) / len(correct)
        val_accuracy.append(epoch_validation_accuracy)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train loss: {epoch_train_loss}, Validation loss: {epoch_validation_loss}, Validation accuracy: {epoch_validation_accuracy}")

        tb_writer.add_scalar("train:loss", epoch_train_loss, epoch)
        tb_writer.add_scalar("validation:loss", epoch_validation_loss, epoch)
        tb_writer.add_scalar("validation:accuracy", epoch_validation_accuracy, epoch)
        tb_writer.flush()

    sns.set()
    plt.plot(train_loss, label="Training")
    plt.plot(val_loss, label="Validation")
    plt.legend()
    plt.ylabel("CE Loss")
    plt.xlabel("Epoch")
    plt.title(f"Loss during training")
    plt.savefig(f"runs/imbb_MLP_{timestamp}.png")

    with open(f"runs/imbb_MLP_{timestamp}.txt", "w") as f:
        f.write(f"Final Accuracy: {val_accuracy[-1]*100:.2f}")


if __name__ == "__main__":
    main()