from torch.utils.data import DataLoader

from src.ts_transfer.models.toy import custom_toy_dataset

# generate data
train_dataloader = DataLoader(custom_toy_dataset(seed=42, n=1000, noise_std=0.1), batch_size=10, shuffle=True)
test_dataloader = DataLoader(custom_toy_dataset(seed=42, n=1000, noise_std=0.1), batch_size=10, shuffle=True)



# train model

