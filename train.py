# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_generation import generate_dataset
from model import PRPredictor


# 训练循环
def trainer(model, train_loader, loss_fn, optimizer, num_epochs, *data):
    Xtrain_tensor, Ytrain_tensor, Xtest_tensor, Ytest_tensor = data
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