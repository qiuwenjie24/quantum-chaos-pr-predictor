# 配置文件，实验参数

import torch

class Config:
    # 数据参数
    N_TRAIN = 1000
    N_TEST = 100
    T = 500
    L = 256
    WINDOW_SIZE = 10
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
config = Config()