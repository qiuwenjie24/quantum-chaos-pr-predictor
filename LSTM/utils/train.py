# 训练工具 utils/train.py

import torch
from torch.utils.data import DataLoader

def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs, device):
    """
    训练模型并返回训练和测试损失
    """
    train_loss, test_loss = [], []
    
    # 获取完整数据集
    X_train_all, Y_train_all = train_loader.dataset.tensors
    X_test_all, Y_test_all = test_loader.dataset.tensors
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            train_pred, _ = model(X_train_all.to(device))
            test_pred, _ = model(X_test_all.to(device))
            train_l = loss_fn(train_pred, Y_train_all.to(device)).item()
            test_l = loss_fn(test_pred, Y_test_all.to(device)).item()
        
        train_loss.append(train_l)
        test_loss.append(test_l)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_l:.6f}, Test Loss: {test_l:.6f}")
    
    return train_loss, test_loss