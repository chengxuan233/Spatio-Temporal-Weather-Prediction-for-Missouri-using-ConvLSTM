import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# === 加载评估结果 ===
df = pd.read_csv("eval_metrics_per_channel.csv")

# === 绘制 F1 分数时间序列图 ===
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
print("✅ Saved: f1_score_over_time.png")
plt.show()

# === 构建总体混淆矩阵 ===
# 加载掩码（你可以用 numpy 保存的完整预测掩码替代）
true_all = []
pred_all = []

# 用 csv 中 Precision/Recall/F1 推测 TP/FP/FN，总体构造掩码
for p, r, f1 in zip(df["Precision_channel_0"], df["Recall_channel_0"], df["F1_channel_0"]):
    if np.isnan(f1):
        continue
    TP = f1 * (p + r) / (2 * p) * 100  # 粗略放大系数
    FN = TP / r - TP
    FP = TP / p - TP
    TN = 10000 - TP - FP - FN  # 假设总格点为10000个
    true_all += [1] * int(TP + FN) + [0] * int(FP + TN)
    pred_all += [1] * int(TP + FP) + [0] * int(FN + TN)

# 限制数量一致
N = min(len(true_all), len(pred_all))
true_all = true_all[:N]
pred_all = pred_all[:N]

# 绘制混淆矩阵
cm = confusion_matrix(true_all, pred_all)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Rain", "Rain"])
plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Binary Confusion Matrix (Channel 0)")
plt.savefig("confusion_matrix_precip.png")
print("✅ Saved: confusion_matrix_precip.png")
plt.show()
