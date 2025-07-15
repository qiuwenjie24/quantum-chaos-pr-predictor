# 数据模块 data/generate.py

import numpy as np
# from scipy.fft import fft, ifft

def kicked_rotor_pr(K, hbar, T=500, L=256):
    theta = np.linspace(0, 2*np.pi, L, endpoint=False)
    n = np.fft.fftfreq(L, d=2*np.pi/L)  # 无量纲动量
    psi = np.exp(-((theta - np.pi)**2)/(2 * 0.3**2))
    psi /= np.linalg.norm(psi)

    PRs = []
    for _ in range(T):
        psi = np.exp(-1j * K * np.cos(theta)) * psi  # 踢：位置基
        psi_n = np.fft.fft(psi)
        psi_n *= np.exp(-1j * hbar * (n**2))   # 动能：动量基
        psi = np.fft.ifft(psi_n)
        prob = np.abs(psi_n)**2
        PR = (np.sum(prob)**2) / np.sum(prob**2)
        PRs.append(PR.real)
    return np.array(PRs)

def generate_dataset(N, T, L):
    X, Y = [], []
    for _ in range(N):
        K = np.random.uniform(0.5, 10.0)
        hbar = np.random.uniform(0.1, 2.0)
        pr_seq = kicked_rotor_pr(K, hbar, T, L)
        X.append([K, hbar])
        Y.append(pr_seq)
    return np.array(X), np.array(Y)
