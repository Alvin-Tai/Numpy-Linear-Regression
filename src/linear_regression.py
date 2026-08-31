"""
线性回归模型 - 纯NumPy实现
包含：批量梯度下降训练、MSE损失计算、R^2评估
"""

import numpy as np


class LinearRegression:
    """
    多元线性回归模型
    使用批量梯度下降（Batch Gradient Descent）优化
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化模型参数

        Parameters:
        -----------
        learning_rate : float
            学习率 α，控制梯度下降的步长
        n_iterations : int
            梯度下降的迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # 权重向量 w
        self.bias = None  # 偏置项 b
        self.loss_history = []  # 记录每次迭代的损失值，用于可视化

    def fit(self, X, y):
        """
        训练模型 - 批量梯度下降

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            训练数据，m个样本，n个特征
        y : np.ndarray, shape (m,)
            目标值
        """
        m, n = X.shape  # m: 样本数, n: 特征数

        # 初始化参数：权重为零向量，偏置为零
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        # 梯度下降迭代
        for i in range(self.n_iterations):
            # 1. 计算当前预测值：h(x) = w^T * x + b
            y_pred = self.predict(X)

            # 2. 计算损失（MSE）
            loss = self._compute_loss(y, y_pred, m)
            self.loss_history.append(loss)

            # 3. 计算梯度
            # dw = (1/m) * X^T * (y_pred - y)
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            # db = (1/m) * sum(y_pred - y)
            db = (1 / m) * np.sum(y_pred - y)

            # 4. 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 每100次迭代打印一次损失
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.6f}")

    def predict(self, X):
        """
        预测

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            输入数据

        Returns:
        --------
        y_pred : np.ndarray, shape (m,)
            预测值
        """
        return np.dot(X, self.weights) + self.bias

    def _compute_loss(self, y_true, y_pred, m):
        """
        计算均方误差（MSE）
        J = (1 / 2m) * Σ(y_pred - y_true)^2
        前面乘1/2是为了求导时系数更简洁
        """
        return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

    def score(self, X, y):
        """
        计算R^2决定系数（模型拟合优度）
        R^2 = 1 - SS_res / SS_tot
        越接近1说明拟合越好
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)  # 残差平方和
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        return 1 - (ss_res / ss_tot)

    def get_params(self):
        """返回训练后的参数"""
        return {"weights": self.weights, "bias": self.bias}
