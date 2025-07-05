# main 主程序入口

from data_generation import generate_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import PRPredictor
from train import trainer
import matplotlib.pyplot as plt

# 超参数设置
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

# 训练并保存
train_loss, test_loss = trainer(model, train_loader, loss_fn, optimizer, num_epochs, 
							Xtrain_tensor, Ytrain_tensor, Xtest_tensor, Ytest_tensor)
torch.save(model.state_dict(), 'model.pth')

# 可视化损失
plt.figure() 
plt.plot(range(1, num_epochs + 1), train_loss, label='train', linestyle='-', color='blue')
plt.plot(range(1, num_epochs + 1), test_loss, label='test', linestyle='-', color='red')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title('Loss')
plt.grid(True)
plt.savefig('Train_Loss.png')
plt.show()
