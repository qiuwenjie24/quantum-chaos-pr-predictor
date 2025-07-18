# data_generation.py
import numpy as np

# 根据公式计算 PR 序列 
def kicked_rotor_pr(K, hbar, T=100, L=256):
    x = np.linspace(0, 2*np.pi, L, endpoint=False)
    p = np.fft.fftfreq(L, d=2*np.pi/L) * 2 * np.pi
    psi = np.exp(-((x - np.pi)**2) / (2 * 0.3**2))
    psi /= np.linalg.norm(psi)

    PRs = []
    for _ in range(T):
        psi = np.exp(-1j * K * np.cos(x) / hbar) * psi
        psi_p = fft(psi)
        psi_p = np.exp(-1j * (p**2) / (2 * hbar)) * psi_p
        psi = ifft(psi_p)
        prob = np.abs(fft(psi))**2
        PR = (np.sum(prob)**2) / np.sum(prob**2)
        PRs.append(PR.real)

    return np.array(PRs)

# 基于物理模型生成 PR 数据 
def generate_dataset(N=1000, T=100):
    X, Y = [], []
    for _ in range(N):
        K = np.random.uniform(0.5, 10.0)
        hbar = np.random.uniform(0.1, 2.0)
        pr_seq = kicked_rotor_pr(K, hbar, T)
        X.append([K, hbar])
        Y.append(pr_seq)
    return np.array(X), np.array(Y)
