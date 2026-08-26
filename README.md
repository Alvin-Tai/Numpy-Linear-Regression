# NumPy Linear Regression

用 NumPy 实现多元线性回归（Batch Gradient Descent），包含完整的训练、可视化与评估流程。

## 项目结构

```text
numpy-linear-regression/
├── data/               # 数据集
├── src/                # 源代码
├── figures/            # 可视化结果
├── main.py             # 入口文件
└── requirements.txt    # 依赖
```

## 环境要求

- Python >= 3.8
- NumPy
- Matplotlib
- Pandas

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行
python main.py

运行后会：
- 生成模拟数据集并保存到 data/synthetic_data.csv
- 训练线性回归模型（批量梯度下降）
- 在 figures/ 目录下生成三张可视化图片：
  - data_distribution.png：数据分布
  - training_loss.png：训练损失下降曲线
  - regression_result.png：回归拟合效果


## 核心公式
H(x) = w^T * x + b

Loss(w,b) = (1 / 2m) * Σ(h(x_i) - y_i)^2

GD
w := w - α * (1/m) * Σ(h(x_i) - y_i) * x_i
b := b - α * (1/m) * Σ(h(x_i) - y_i)


## 作者
Alvin-Tai from BUCT

