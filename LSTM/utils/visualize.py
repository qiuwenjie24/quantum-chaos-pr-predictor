# 可视化和预测 utils/visualize.py

import matplotlib.pyplot as plt
import numpy as np
from data.generate import kicked_rotor_pr
import torch

def plot_results(train_loss, test_loss, epochs):
    """绘制训练和测试损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

def plot_prediction(model, config, sample_params, true_sequence):
    """绘制预测结果与真实序列的对比"""
    model.eval()
    device = config.DEVICE
    window_size = config.WINDOW_SIZE
    
    # 生成真实序列
    K, hbar = sample_params
    true_seq = kicked_rotor_pr(K, hbar, config.T, config.L)
    
    # 初始化预测
    pred_seq = list(true_seq[:window_size])
    h = torch.zeros(2, 1, model.lstm.hidden_size).to(device)
    c = torch.zeros(2, 1, model.lstm.hidden_size).to(device)
    
    # 逐步预测
    with torch.no_grad():
        for _ in range(config.T - window_size):
            input_data = np.concatenate([[K, hbar], pred_seq[-window_size:]])
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            pred, (h, c) = model(input_tensor, (h, c))
            pred_seq.append(pred.item())
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(true_seq, label='True Sequence', linewidth=2)
    plt.plot(pred_seq, label='Predicted Sequence', linestyle='--')
    plt.axvline(x=window_size, color='r', linestyle='--', label='Prediction Start')
    plt.xlabel('Time Step')
    plt.ylabel('PR Value')
    plt.title(f'Prediction vs Ground Truth (K={K:.2f}, ħ={hbar:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison.png')
    plt.show()