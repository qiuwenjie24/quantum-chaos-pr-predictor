# 模型定义 models/lstm.py

import torch.nn as nn
import torch

class PRPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, window_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.params_fc = nn.Linear(input_dim, hidden_dim)  # 额外处理K和hbar
        self.output_fc = nn.Linear(hidden_dim * 2, 1)  # 合并

    def forward(self, x, hidden=None):
        params = x[:, :2]
        pr_history = x[:, 2:].unsqueeze(-1)
        
        if hidden is not None:
            lstm_out, hidden_out = self.lstm(pr_history, hidden)
        else:
            lstm_out, hidden_out = self.lstm(pr_history)
            
        lstm_last = lstm_out[:, -1, :]
        params_feat = self.params_fc(params)
        
        combined = torch.cat([lstm_last, params_feat], dim=1)
        output = self.output_fc(combined).squeeze(-1)
        return output, hidden_out