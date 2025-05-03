import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# === Load Evaluation Results ===
df = pd.read_csv("eval_metrics_per_channel.csv")

# === Plot F1 Score Time Series for Precipitation Channel (Channel 0) ===
plt.figure(figsize=(10, 4))
plt.plot(df["Index"], df["F1_channel_0"], label="F1 Score (Precipitation)", color="darkblue")
plt.xlabel("Time Step")
plt.ylabel("F1 Score")
plt.title("F1 Score Over Time (Channel 0)")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("f1_score_over_time.png")
print("Saved: f1_score_over_time.png")
plt.show()

# === Construct Approximate Global Confusion Matrix ===
# Note: For true accuracy, use saved prediction masks directly
true_all = []
pred_all = []

# Estimate TP/FP/FN based on precision/recall/F1 (approximate, scaled)
for p, r, f1 in zip(df["Precision_channel_0"], df["Recall_channel_0"], df["F1_channel_0"]):
    if np.isnan(f1):
        continue
    TP = f1 * (p + r) / (2 * p) * 100  # Scaled factor for illustration
    FN = TP / r - TP
    FP = TP / p - TP
    TN = 10000 - TP - FP - FN  # Assume total grid points per frame is 10,000
    true_all += [1] * int(TP + FN) + [0] * int(FP + TN)
    pred_all += [1] * int(TP + FP) + [0] * int(FN + TN)

# Ensure equal lengths for input to confusion matrix
N = min(len(true_all), len(pred_all))
true_all = true_all[:N]
pred_all = pred_all[:N]

# === Plot Confusion Matrix ===
cm = confusion_matrix(true_all, pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Binary Confusion Matrix (Channel 0)")
plt.savefig("confusion_matrix_precip.png")
print("Saved: confusion_matrix_precip.png")
plt.show()
