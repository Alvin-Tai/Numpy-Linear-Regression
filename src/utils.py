"""
工具函数：数据生成、数据加载、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_synthetic_data(n_samples=100, n_features=1, noise=10.0, seed=42):
    """
    生成模拟数据集：y = 3*x1 + 5*x2 + ... + 10 + noise

    Parameters:
    -----------
    n_samples : int
        样本数量
    n_features : int
        特征数量（这里默认1维，方便可视化）
    noise : float
        噪声强度
    seed : int
        随机种子，保证可复现

    Returns:
    --------
    X : np.ndarray
        特征矩阵
    y : np.ndarray
        目标值
    true_weights : np.ndarray
        真实的权重（用于对比模型学习效果）
    true_bias : float
        真实的偏置
    """
    np.random.seed(seed)

    # 生成随机特征
    X = np.random.randn(n_samples, n_features) * 10  # 放大范围，数据更有趣

    # 设定真实参数
    true_weights = np.array([3.0] * n_features)  # 真实权重为3
    true_bias = 10.0  # 真实偏置为10

    # 生成目标值：y = w^T * x + b + noise
    y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples) * noise

    return X, y, true_weights, true_bias


def save_data(X, y, filepath="data/synthetic_data.csv"):
    """
    将数据保存为CSV文件
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    df["target"] = y
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def plot_data_distribution(X, y, save_path="figures/data_distribution.png"):
    """
    绘制数据分布散点图（仅支持1维特征）
    """
    if X.shape[1] != 1:
        print("Data distribution plot only supports 1D feature.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, alpha=0.6, color="blue", label="Data points")
    plt.xlabel("Feature (x)", fontsize=12)
    plt.ylabel("Target (y)", fontsize=12)
    plt.title("Synthetic Data Distribution", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Data distribution plot saved to {save_path}")


def plot_training_loss(loss_history, save_path="figures/training_loss.png"):
    """
    绘制训练过程中的损失下降曲线
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="red", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Training Loss Curve (Gradient Descent)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training loss plot saved to {save_path}")


def plot_regression_result(X, y, model, true_weights, true_bias, save_path="figures/regression_result.png"):
    """
    绘制回归拟合结果：真实直线 vs 模型预测直线（仅支持1维特征）
    """
    if X.shape[1] != 1:
        print("Regression result plot only supports 1D feature.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 生成平滑的x值用于画线
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)

    # 模型预测线
    y_pred_line = model.predict(x_line)

    # 真实线
    y_true_line = true_weights[0] * x_line[:, 0] + true_bias

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, alpha=0.6, color="blue", label="Data points")
    plt.plot(x_line[:, 0], y_pred_line, color="red", linewidth=2, label=f"Learned: y = {model.weights[0]:.2f}x + {model.bias:.2f}")
    plt.plot(x_line[:, 0], y_true_line, color="green", linewidth=2, linestyle="--", label=f"True: y = {true_weights[0]:.2f}x + {true_bias:.2f}")
    plt.xlabel("Feature (x)", fontsize=12)
    plt.ylabel("Target (y)", fontsize=12)
    plt.title("Linear Regression Result", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Regression result plot saved to {save_path}")
