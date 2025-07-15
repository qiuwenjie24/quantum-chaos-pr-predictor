# 主程序 main.py

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from config import Config
from data.generate import generate_dataset
from data.preprocess import create_sequential_dataset
from models.lstm import PRPredictor
from utils.train import train_model
from utils.visualize import plot_results, plot_prediction

def main():
    cfg = Config()
    print(f"Using device: {cfg.DEVICE}")
    
    # 1. 数据生成
    print("Generating dataset...")
    X_train, Y_train = generate_dataset(cfg.N_TRAIN, cfg.T, cfg.L)
    X_test, Y_test = generate_dataset(cfg.N_TEST, cfg.T, cfg.L)
    
    # 2. 数据预处理
    print("Preprocessing data...")
    X_train_seq, Y_train_seq = create_sequential_dataset(X_train, Y_train, cfg.WINDOW_SIZE)
    X_test_seq, Y_test_seq = create_sequential_dataset(X_test, Y_test, cfg.WINDOW_SIZE)
    
    # 3. 创建数据加载器
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(Y_train_seq, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(Y_test_seq, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    model = PRPredictor(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        window_size=cfg.WINDOW_SIZE
    ).to(cfg.DEVICE)
    
    # 5. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.MSELoss()
    
    # 6. 训练模型
    print("Starting training...")
    train_loss, test_loss = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=cfg.EPOCHS,
        device=cfg.DEVICE
    )
    
    # 7. 可视化训练结果
    plot_results(train_loss, test_loss, cfg.EPOCHS)
    
    # 8. 生成预测结果
    plot_prediction(model, cfg, X_train[0], Y_train[0])

if __name__ == "__main__":
    main()