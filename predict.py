# predict.py
import torch
import matplotlib.pyplot as plt
from model import PRPredictor
from data_generation import kicked_rotor_pr

model = PRPredictor(T=50)
model.load_state_dict(torch.load("model.pt"))
model.eval()

with torch.no_grad():
    test_K, test_hbar = 4.3, 0.7
    test_x = torch.tensor([[test_K, test_hbar]], dtype=torch.float32)
    pred_seq = model(test_x).numpy()[0]
    true_seq = kicked_rotor_pr(test_K, test_hbar, T=50)

plt.plot(pred_seq, label="Predicted PR")
plt.plot(true_seq, label="True PR")
plt.xlabel("Time")
plt.ylabel("Participation Ratio")
plt.legend()
plt.title("QKR PR Prediction vs Ground Truth")
plt.show()