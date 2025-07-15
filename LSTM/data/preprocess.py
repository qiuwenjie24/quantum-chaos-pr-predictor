# 数据预处理 data/preprocess.py

import numpy as np

# 生成适配LSTM的数据集
def create_sequential_dataset(X_params, Y_sequences, window_size):
    inputs, targets = [], []
    for i in range(len(X_params)):
        for t in range(window_size, len(Y_sequences[i])):
            inputs.append(np.concatenate([X_params[i], Y_sequences[i][t-window_size:t]]))
            targets.append(Y_sequences[i][t])
    return np.array(inputs), np.array(targets)
