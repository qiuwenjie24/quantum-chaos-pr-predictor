# 基于神经网络的量子混沌系统动力学预测

## 文件结构

quantum-chaos-pr-predictor/

├── data_generation.py      <- QKR 模拟与数据生成模块

├── model.py                    <- 神经网络模型定义

├── train.py                       <- 模型训练逻辑

├── predict.py                   <- 单个样本预测与可视化

├── requirements.txt          <- 所需依赖库

└── README.md               <- 项目介绍（报告内容）

## 一、项目背景与动机

量子混沌系统广泛存在于原子物理、量子信息、凝聚态等多个领域，其动力学行为通常表现出非线性、高维和强耦合等复杂特征。传统的数值模拟虽然精确，但计算成本高，不具备泛化能力。

本项目以典型的量子混沌模型——量子踢转子（Quantum Kicked Rotor, QKR）为研究对象，尝试通过神经网络模型学习系统参数（踢强度 $K$ 与有效普朗克常数 $\hbar$）与动力学演化物理量（如参与比 PR）的映射关系，从而建立一种数据驱动的、可高效预测动力学行为的模型。

## 二、研究目标

构建一个监督学习模型，输入系统参数 $[K, \hbar]$，输出该参数下系统参与比随时间变化的序列：$
[PR(t=0), PR(t=1), \dots, PR(t=T)]$，实现对量子混沌动力学的端到端预测。

## 三、方法与实现

### 3.1 模型选择

采用多层前馈神经网络（MLP）作为基础模型，其结构如下：

- 输入层：2维（系统参数 $K$ 和 $\hbar$）
- 隐含层：两层 ReLU 激活（$64 \to 128$）
- 输出层：T维，预测 T 个时间步的参与比值

### 3.2 数据生成

使用 QKR 模型的数值模拟生成训练数据：

- 初态为高斯波包
- 采用 Floquet 算符推进波函数演化
- 在动量表象下计算概率密度，进而计算参与比：$PR(t) = \frac{\left(\sum_l |\psi_l|^2\right)^2}{\sum_l |\psi_l|^4}$

**数据集：** 共生成 1000 个样本，每个样本对应一个参数对 $[K, \hbar]$ 和一条长度为 T=50 的参与比时间序列。



**理论计算**：

设 $\{|p_j\rangle\}$ 为角动量 $L$ 的本征态，$\{x_i\}$为角度 $\theta$ 的本征态。

量子态在动量本征态下的表示为， $|\psi\rang=\sum_j \psi_j|p_j\rangle$，其中 $\psi_j=\langle p_j| \psi\rangle$ 是我们想要得到的。

记初始量子态为 $|\psi(0)\rangle$，在 QKT 系统下的时间演化为：
$$
|\psi(t+T)\rangle = \mathrm{e}^{-\text{i}\frac{L^2}{2\hbar}T }\cdot \mathrm{e}^{-\text{i}\frac{K}{\hbar}\cos\theta } |\psi(t)\rangle
$$
则$\psi_k(t+T)=\langle p_k| \psi(t+T)\rangle$ 为：
$$
\begin{align}
\langle p_k|\psi(t+T)\rangle &= \sum_{i,j }\langle p_k| \mathrm{e}^{-\text{i}\frac{L^2}{2\hbar}T }|p_j\rangle \langle p_j|\mathrm{e}^{-\text{i}\frac{K}{\hbar}\cos\theta } |x_i\rangle \langle x_i |\psi(t)\rangle \\
&=  \sum_{i,j } \mathrm{e}^{-\text{i}\frac{L_k^2}{2\hbar}T } \mathrm{e}^{-\text{i}\frac{K}{\hbar}\cos\theta_i }  \langle p_k |p_j\rangle \langle p_j|x_i\rangle \langle x_i |\psi(t)\rangle \\
&= \sum_{i} \mathrm{e}^{-\text{i}\frac{L_k^2}{2\hbar}T } \mathrm{e}^{-\text{i}\frac{K}{\hbar}\cos\theta_i }  \frac{1}{\sqrt{2\pi \hbar}}\mathrm{e}^{\text{i}\frac{L_k \theta_i} {\hbar} } \langle x_i |\psi(t)\rangle
\end{align}
$$






### 3.3 模型训练

- 损失函数：均方误差（MSE）
- 优化器：Adam，学习率 1e-3
- 批大小：32
- 训练轮数：10（可调）

模型在验证集上表现出良好的预测能力，平均 MSE 低于 0.01，且预测曲线与真实参与比曲线拟合良好。



### 3.4 可视化示例

对未见过的参数对 $(K=4.3, \hbar=0.7)$，模型预测的 PR 曲线与真实值几乎重合，说明模型具有一定的泛化能力。

## 四、项目亮点与创新点

1. **物理建模与机器学习融合**：通过数据驱动方式近似传统演化模型，展示了 AI 技术在复杂物理系统中的应用潜力。
2. **无需外部数据集**：全部样本基于自定义物理模型生成，适合教学、科研与工程快速验证。
3. **简洁有效的结构设计**：使用 MLP 实现端到端预测，结构简洁，训练高效。

## 五、后续工作

- 引入时序模型（如 LSTM 或 Transformer）模拟逐步演化过程
- 增加初始态多样性，提高模型鲁棒性
- 引入混沌/非混沌标签，进行分类建模探索
- 尝试逆问题建模：由 PR 曲线预测系统参数

## 六、总结

本项目实现了从物理参数到量子混沌动力学行为的神经网络预测框架，展示了深度学习在复杂量子系统建模中的潜力。该工作不仅有助于理解深度学习与物理系统的结合方式，也为进一步的 AI + 物理交叉研究提供了实验平台。

---

**项目周期**：4 周  

**开发工具**：Python, NumPy, SciPy, PyTorch, Matplotlib  

