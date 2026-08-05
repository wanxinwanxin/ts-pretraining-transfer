from torch.utils.data import DataLoader
from ts_transfer.data.toy import ToyDataset
from ts_transfer.models.toy import TwoLayerMLP
import torch.optim as optim
import torch.nn as nn
import torch


# generate data
train_dataloader = DataLoader(ToyDataset(seed=42, n=1000, noise_std=0.1), batch_size=10, shuffle=True)
test_dataloader = DataLoader(ToyDataset(seed=142, n=1000, noise_std=0.1,), batch_size=10, shuffle=False)

torch.manual_seed(42)
model = TwoLayerMLP(input_size=1, hidden_size=32, output_size=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


# train model
for epoch in range(200):
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
    



