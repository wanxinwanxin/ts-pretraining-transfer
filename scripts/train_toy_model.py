import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ts_transfer.data.toy import ToyDataset
from ts_transfer.models.toy import TwoLayerMLP

os.makedirs("results/toy", exist_ok=True)

# implement a --resume flag to load the checkpoint
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

torch.manual_seed(42)
model = TwoLayerMLP(input_size=1, hidden_size=32, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
start_epoch = 0
end_epoch = 200

if args.resume:
    checkpoint = torch.load("results/toy/checkpoint.pt")
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    optimizer.load_state_dict(state_dict=checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Resumed from epoch {epoch}, loss {loss}")
    start_epoch = epoch + 1
    end_epoch = epoch + 200

# prepare data
train_dataloader = DataLoader(
    ToyDataset(seed=42, n=1000, noise_std=0.1), batch_size=10, shuffle=True
)
test_dataloader = DataLoader(
    ToyDataset(
        seed=142,
        n=1000,
        noise_std=0.1,
    ),
    batch_size=10,
    shuffle=False,
)

# train model
for epoch in range(start_epoch, end_epoch):
    total_loss = 0
    for batch in train_dataloader:
        x, y = batch
        y_pred = model(x)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_dataloader)}")

# save and checkpoint (model, optimizer, epoch, loss)
# save to results/toy/checkpoint.pt
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss / len(train_dataloader),
    },
    "results/toy/checkpoint.pt",
)
