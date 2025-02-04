import syft as sy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Create a virtual environment for federated learning
hook = sy.TorchHook(torch)

# Create virtual workers
alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")

# Split training data between workers
def split_data(x, y):
    # Split data evenly between Alice and Bob
    split = len(x) // 2
    x_alice, x_bob = x[:split], x[split:]
    y_alice, y_bob = y[:split], y[split:]
    return x_alice, y_alice, x_bob, y_bob

x_alice, y_alice, x_bob, y_bob = split_data(torch.tensor(x_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long))

# Create federated datasets
federated_train_loader = sy.FederatedDataLoader(TensorDataset(x_alice, y_alice).federate([alice]),
                                                batch_size=64, shuffle=True)

# Define a simple model
model = nn.Sequential(
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# Federated learning training loop
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# After training, gather model weights from workers
model_weights = model.get()
