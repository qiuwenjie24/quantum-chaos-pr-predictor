# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_generation import generate_dataset
from model import PRPredictor

X_np, Y_np = generate_dataset(N=1000, T=50)
X_tensor = torch.tensor(X_np, dtype=torch.float32)
Y_tensor = torch.tensor(Y_np, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=32, shuffle=True)

model = PRPredictor(T=50)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(10):
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pt")