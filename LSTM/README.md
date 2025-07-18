# 基于神经网络的量子混沌系统动力学预测(LSTM版)

## 文件结构

```text
quantum-chaos-pr-predictor/
│
├── data/
│   ├── generate.py
│   └── preprocess.py
│
├── models/
│   └── lstm.py
│
├── utils/
│   ├── train.py
│   └── visualize.py
│
├── config.py
├── main.py
├── .gitignore
├── requirements.txt
└── README.md
```



## 研究背景

原先是使用MLP模型处理：

- **目标**：已知系统参数 `[K, hbar]` 预测 PR 序列 `[PR(1),...,PR(T)]`，模型直接学习参数到序列的全局映射。
- **模型输入**：系统参数 `[K, hbar]` 
- **模型输出**：PR 序列 `[PR(1),...,PR(T)]` 

由于PR随时间的演化表现出长时记忆和非线性特征，而LSTM在处理这类序列中具有良好表现，避免梯度消失，能更稳定地学习PR历史对未来的影响。因此这里将MLP改进成LSTM。



## 研究目标

假设**初始条件**：

- 系统参数：`[K, hbar]`
- 已知前 n 个PR值：`[PR(1),...,PR(n)`（即"窗口"`window_size=n`）

**目标**：通过逐步预测（自回归）生成完整序列 `[PR(1),...,PR(T)`，模型学习局部时序演化规则，实现对量子混沌动力学的端到端预测。



## 模型架构

**模型输入**：`[K, hbar, PR(t-1), PR(t-2), ..., PR(t-n)]`

**模型输出**：`PR_t`

**数学公式**：
$$
PR(n+k) = F(K,\hbar, PR(k),\cdots, PR(n+k-1)),\quad k=1,\cdots d \nonumber
$$


**设计思想图解**：

```
输入: [K, hbar, PR_t-1, PR_t-2, ..., PR_t-n] 
      │           │
      │           └──────────────┐
      ↓                          ↓
[全连接层]                     [LSTM]
   ↓                            ↓
(64维参数特征)            (64维序列特征)
      └───────────[拼接]──────────┘
                         │
                     [全连接层]
                         ↓
                     预测PR_t
```

 注：`K` 和 `hbar` 需要单独处理，因为它们是决定PR序列演化的**全局参数**，与局部时序特征（窗口PR值）具有不同物理意义。分开处理能让模型更明确地学习两者的影响。



## 方法与实现

### 3.1 数据处理

这里模型的输入是系统参数 `K,hbar` 和前 $n$ 个时刻的 PR 值，不同于MLP模型的情况，因此，需要重新构造数据集。

特征：`[K, hbar, pr(i), ..., pr(i+n-1)]`

标签：`pr(n)`



### **3.2 训练**

训练时通常不保留状态，每个batch独立初始化（默认全零初始化），防止梯度跨batch传播导致的不稳定。

注意：若想训练时也保持状态连续性（如处理超长序列），可用 `torch.utils.rnn.PackedSequence` 和状态截断（truncated BPTT），但会显著增加实现复杂度，这里不采用这种。



### 3.3 预测

预测时，通过滑动窗口，逐步生成完整序列。

因为预测是自回归预测（用预测值作为下一步输入），则需要存储状态，手动传递上一时刻的状态 (`h_n,c_n`)，实现连续预测。



### 3.4 新增

使用了GPU并行运算，加快训练速度。



## 优势

相比原先MLP模型的优势：

1. **物理规律**

   PR序列的演化由参数（K, hbar）和系统历史状态共同决定，是一个典型的时间依赖过程。LSTM通过门控机制能显式建模长期依赖关系，相比MLP的直接映射，LSTM更符合物理规律。

2. **参数整合**

   K和hbar作为全局控制参数，决定了PR序列的整体行为模式（如周期性、混沌性），而窗口内的PR历史值提供局部动态信息。两者结合能同时捕捉宏观和微观规律。

3. **自回归的可行性**

   PR序列的演化通常是马尔可夫性的（即下一状态仅依赖有限历史）。通过适当选择窗口大小（如10-20步），LSTM能足够捕捉关键依赖。

4. **信息完整性**

   实验表明，Kicked Rotor模型的PR序列可通过有限历史步数有效预测（相关论文如[Chabé et al., Phys. Rev. E 2019]）。

5. **参数外推**

   在训练集未覆盖的(K, hbar)区域，LSTM比MLP更可能给出物理合理的序列。



## **潜在理论挑战与改进**

1. **窗口大小选择**：

   - 问题：窗口过小会丢失长程依赖，过大增加计算负担。
   - 解决方案：使用注意力机制替代固定窗口（如Transformer）。

2. **参数-时序耦合建模**：

   - 问题：简单拼接K/hbar与PR历史可能限制模型表达能力。
   - 解决方案：将参数作为LSTM的初始隐藏状态（而非输入）。

3. **预处理**：对数据进行标准化，使得模型训练稳定，收敛更快。其中静态/动态特征分开处理，动态序列按样本独立标准化（避免不同样本间尺度差异）。

4. **特征工程**：添加一阶差分特征 (ΔPR = PR_t - PR_t-1)，显式提供LSTM可能难以自动学习的关键模式。

5. 将MLP与LSTM相结合，前n时刻的PR值由MLP预测，后续由LSTM预测，实现仅由系统参数预测完整PR序列。

   








