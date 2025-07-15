import numpy as np
from scipy.fft import fft, ifft
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 原始物理模型部分保持不变
def kicked_rotor_pr(K, hbar, T=500, L=256):
    theta = np.linspace(0, 2*np.pi, L, endpoint=False)
    n = np.fft.fftfreq(L, d=2*np.pi/L)  # 无量纲动量
    psi = np.exp(-((theta - np.pi)**2)/(2 * 0.3**2))
    psi /= np.linalg.norm(psi)

    PRs = []
    for _ in range(T):
        psi = np.exp(-1j * K * np.cos(theta)) * psi       # 踢：位置基
        psi_n = np.fft.fft(psi)
        psi_n *= np.exp(-1j * hbar * (n**2))             # 动能：动量基
        psi = np.fft.ifft(psi_n)
        prob = np.abs(psi_n)**2
        PR = (np.sum(prob)**2) / np.sum(prob**2)
        PRs.append(PR.real)

    return np.array(PRs)

def generate_dataset(N=1000, T=500, L=256):
    X, Y = [], []
    for _ in range(N):
        K = np.random.uniform(0.5, 10.0)
        hbar = np.random.uniform(0.1, 2.0)
        pr_seq = kicked_rotor_pr(K, hbar, T, L)
        X.append([K, hbar])
        Y.append(pr_seq)

    return np.array(X), np.array(Y)

def generate_sequential_dataset(N=1000, T=500, L=256, window_size=10):
    """
    X_params: (N, 2) 的K和hbar
    Y_sequences: (N, T) 的PR序列
    返回: (输入, 目标)
    输入形状: (样本数, 窗口大小+2) = (K, hbar, PR_t-1, PR_t-2, ..., PR_t-window_size)
    目标形状: (样本数,) = PR_t
    """
    X_params, Y_sequences = generate_dataset(N, T, L)
    inputs, targets = [], []
    for i in range(len(X_params)):
        for t in range(window_size, len(Y_sequences[i])):
            inputs.append(np.concatenate([
                X_params[i], 
                Y_sequences[i][t-window_size:t]
            ]))
            targets.append(Y_sequences[i][t])
    return np.array(inputs), np.array(targets)


# LSTM
class PRPredictor(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        # 处理 历史PR（window_size）
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间步输入1个PR值，序列长度=时间步数量，即window_size个时间步
            hidden_size=4,
            num_layers=2,   # 2层，每层4个记忆元
            batch_first=True   # 把批量放到第一位，(batch, seq_len, input_size)，只针对输入和输出，不改变隐状态
        )
        # 额外处理K和hbar
        self.params_fc = nn.Linear(2, 4)
        # 最终预测
        self.output_fc = nn.Linear(4 + 4, 1)  # 合并LSTM输出和参数特征

    def forward(self, x, hidden=None):
        # x形状: (batch_size, 2 + window_size)
        params = x[:, :2]  # K和hbar
        pr_history = x[:, 2:].unsqueeze(-1)  # 在最后增加长度为1的维度，(batch, window_size, 1)
        
        # 处理历史PR序列
        # 输出为output, (h_n, c_n)，形状为(batch, seq_len, hidden_size) (num_layers, batch, hidden_size) (同前一个)
        if hidden is None: # 如果没有传入初始状态，则默认初始化
            lstm_out, hidden_out = self.lstm(pr_history) 
        else:
            # 传入初始状态 hidden = (h, c)
            lstm_out, hidden_out = self.lstm(pr_history, hidden) 

        
        lstm_last = lstm_out[:, -1, :]  # 取最后时间步的输出 (batch, hidden_size)，最后时间步包含了历史PR记忆
        
        # 处理参数
        params_feat = self.params_fc(params)  # (batch, 4)
        
        # 合并特征
        combined = torch.cat([lstm_last, params_feat], dim=1)
        output = self.output_fc(combined).squeeze(-1)  # (batch,)
        return output, hidden_out



# 自动选择设备（优先GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数保持不变
batch_size = 32
num_epochs = 20
lr = 1e-1

# 数据准备与标准化处理
N=100; T=50; L=10; window_size=10 
Xtrain_seq, Ytrain_seq = generate_sequential_dataset(N, T, L, window_size)

Xtrain_tensor = torch.tensor(Xtrain_seq, dtype=torch.float32)
Ytrain_tensor = torch.tensor(Ytrain_seq, dtype=torch.float32)

Xtest_seq, Ytest_seq = generate_sequential_dataset(N, T, L, window_size)

Xtest_tensor = torch.tensor(Xtest_seq, dtype=torch.float32)
Ytest_tensor = torch.tensor(Ytest_seq, dtype=torch.float32)

dataset = TensorDataset(Xtrain_tensor, Ytrain_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# 模型初始化部分只需改类名
model = PRPredictor(window_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# 训练循环完全保持不变
def train(model, train_loader, loss_fn, optimizer, num_epochs):
    train_loss, test_loss = [], []
    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)  # 按批次转移数据到设备，使显卡使用更高效，一次性转移容易显存爆炸
            pred, _ = model(xb)                     #原因是分批传入的时候中间变量，计算下一批时上一批的中间临时变量会释放
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_pred, _ = model(Xtrain_tensor.to(device))
        test_pred, _ = model(Xtest_tensor.to(device))
        train_l = loss_fn(train_pred, Ytrain_tensor.to(device)).item()
        test_l = loss_fn(test_pred, Ytest_tensor.to(device)).item()
        train_loss.append(train_l)
        test_loss.append(test_l)
    return train_loss, test_loss

train_loss, test_loss = train(model, train_loader, loss_fn, optimizer, num_epochs)

# 可视化部分完全保持不变
import matplotlib.pyplot as plt
plt.figure() 
plt.plot(range(1, num_epochs + 1), train_loss, label='train', linestyle='-', color='blue')
plt.plot(range(1, num_epochs + 1), test_loss, label='test', linestyle='-', color='red')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title('Loss---LSTM')  # 仅修改标题中的MLP为LSTM
plt.grid(True)


# 预测
model.eval()
test_K, test_hbar = np.random.uniform(0.5, 10.0), np.random.uniform(0.1, 2.0)
true_seq = kicked_rotor_pr(test_K, test_hbar, T)
pred_seq = list(true_seq[:window_size])

# 初始化状态
h, c = torch.zeros(2, 1, 4).to(device), torch.zeros(2, 1, 4).to(device)  # batch_size=1
with torch.no_grad():
    for _ in range(T-window_size):
        test_x = np.concatenate([[test_K, test_hbar], pred_seq[-window_size:] ])
        test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device) #在维度0处插一个维度,(1, 2 + window_size)
        pred, (h,c) = model(test_x, (h,c))
        pred_seq.append(pred.item())
    pred_seq = np.array(pred_seq)
    

plt.figure() 
plt.plot(pred_seq, label="Predicted PR")
plt.plot(true_seq, label="True PR")
plt.xlabel("Time")
plt.ylabel("Participation Ratio")
plt.legend()
plt.title("QKR PR Prediction vs Ground Truth (LSTM)")  # 添加LSTM标识
plt.show()
