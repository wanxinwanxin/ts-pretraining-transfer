# test the following
# 1. dataset is reproducible under the same seed
# 2. dataset is different under different seeds
# 3. a reloaded model's predictions match the original on the same inputs
# 4. one short training run decreases loss

import torch

from ts_transfer.data.toy import ToyDataset
from ts_transfer.models.toy import TwoLayerMLP


def test_dataset_reproducibility():
    dataset1 = ToyDataset(seed=42, n=1000, noise_std=0.1)
    dataset2 = ToyDataset(seed=42, n=1000, noise_std=0.1)
    assert dataset1.x.shape == dataset2.x.shape
    assert dataset1.y.shape == dataset2.y.shape
    assert torch.equal(dataset1.x, dataset2.x)
    assert torch.equal(dataset1.y, dataset2.y)


def test_dataset_different_seeds():
    dataset1 = ToyDataset(seed=42, n=1000, noise_std=0.1)
    dataset2 = ToyDataset(seed=43, n=1000, noise_std=0.1)
    assert not torch.equal(dataset1.x, dataset2.x)
    assert not torch.equal(dataset1.y, dataset2.y)


def test_model_prediction_reproducibility(tmp_path):

    fresh_model = TwoLayerMLP(input_size=1, hidden_size=32, output_size=1)
    fresh_model.eval()
    dataset = ToyDataset(seed=42, n=1000, noise_std=0.1)
    fresh_preds = fresh_model(dataset.x)
    torch.save(
        {
            "model_state_dict": fresh_model.state_dict(),
        },
        tmp_path / "checkpoint.pt",
    )
    checkpoint = torch.load(tmp_path / "checkpoint.pt")

    reloaded_model = TwoLayerMLP(input_size=1, hidden_size=32, output_size=1)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_model.eval()
    reloaded_preds = reloaded_model(dataset.x)
    assert torch.allclose(fresh_preds, reloaded_preds, atol=1e-6)


def test_one_short_training_run_decreases_loss():
    model = TwoLayerMLP(input_size=1, hidden_size=32, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = ToyDataset(seed=42, n=1000, noise_std=0.1)
    loss_fn = torch.nn.MSELoss()
    loss_history = []
    for epoch in range(10):
        y_pred = model(dataset.x)
        loss = loss_fn(y_pred, dataset.y)
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss_history[-1] < loss_history[0]
