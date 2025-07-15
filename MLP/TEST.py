
import numpy as np
from scipy.fft import fft, ifft

def kicked_rotor_pr(K, h_eff, T=100, L=256):
    theta = np.linspace(0, 2*np.pi, L, endpoint=False)
    n = np.fft.fftfreq(L, d=2*np.pi/L)  # 无量纲动量
    psi = np.exp(-((theta - np.pi)**2)/(2 * 0.3**2))
    psi /= np.linalg.norm(psi)

    PRs = []
    for _ in range(T):
        psi = np.exp(-1j * K * np.cos(theta)) * psi       # 踢：位置基
        psi_n = np.fft.fft(psi)
        psi_n *= np.exp(-1j * h_eff * (n**2))             # 动能：动量基
        psi = np.fft.ifft(psi_n)
        prob = np.abs(psi_n)**2
        PR = (np.sum(prob)**2) / np.sum(prob**2)
        PRs.append(PR.real)

    return np.array(PRs)


def generate_dataset(N=1000, T=100):
    X, Y = [], []
    for _ in range(N):
        K = np.random.uniform(0.5, 10.0)
        hbar = np.random.uniform(0.1, 2.0)
        pr_seq = kicked_rotor_pr(K, hbar, T)
        X.append([K, hbar])
        Y.append(pr_seq)

    return np.array(X), np.array(Y)



# ======================================================================
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class PRPredictor(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, T)
        )

    def forward(self, x):
        return self.model(x)


# 超参数
batch_size = 32
num_epochs = 200
lr = 1e-1


# 数据准备
Xtrain_np, Ytrain_np = generate_dataset(N=1000, T=100)
Xtrain_tensor = torch.tensor(Xtrain_np, dtype=torch.float32)
Ytrain_tensor = torch.tensor(Ytrain_np, dtype=torch.float32)

Xtest_np, Ytest_np = generate_dataset(N=100, T=100)
Xtest_tensor = torch.tensor(Xtest_np, dtype=torch.float32)
Ytest_tensor = torch.tensor(Ytest_np, dtype=torch.float32)

dataset = TensorDataset(Xtrain_tensor, Ytrain_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型初始化
model = PRPredictor(T=100)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# 训练循环
def train(model, train_loader, loss_fn, optimizer, num_epochs):
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_l = loss_fn(model(Xtrain_tensor), Ytrain_tensor).item()
        test_l = loss_fn(model(Xtest_tensor), Ytest_tensor).item()
        train_loss.append(train_l)
        test_loss.append(test_l)

    return train_loss, test_loss

train_loss, test_loss = train(model, train_loader, loss_fn, optimizer, num_epochs)



# 预测和可视化示例
import matplotlib.pyplot as plt
plt.figure() 
plt.plot(range(1, num_epochs + 1), train_loss, label='train', linestyle='-', color='blue')
plt.plot(range(1, num_epochs + 1), test_loss, label='test', linestyle='-', color='red')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title('Loss---MLP')
plt.grid(True)


with torch.no_grad():
    test_K, test_hbar = np.random.uniform(0.5, 10.0), np.random.uniform(0.1, 2.0)
    test_x = torch.tensor([[test_K, test_hbar]], dtype=torch.float32)
    pred_seq = model(test_x).numpy()[0]
    true_seq = kicked_rotor_pr(test_K, test_hbar, T=100)

plt.figure() 
plt.plot(pred_seq, label="Predicted PR")
plt.plot(true_seq, label="True PR")
plt.xlabel("Time")
plt.ylabel("Participation Ratio")
plt.legend()
plt.title("QKR PR Prediction vs Ground Truth")
# plt.show()
