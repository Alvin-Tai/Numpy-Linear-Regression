"""
训练流程封装
"""

from .linear_regression import LinearRegression
from .utils import generate_synthetic_data, save_data, plot_data_distribution, plot_training_loss, plot_regression_result


def train_and_evaluate():
    """
    完整的训练与评估流程
    """
    print("=" * 50)
    print("NumPy Linear Regression - Training Pipeline")
    print("=" * 50)

    # 1. 生成模拟数据
    print("\n[1/5] Generating synthetic data...")
    X, y, true_weights, true_bias = generate_synthetic_data(n_samples=100, n_features=1, noise=10.0, seed=42)  # 1维特征，方便可视化
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   True parameters: weights={true_weights}, bias={true_bias}")

    # 2. 保存数据
    print("\n[2/5] Saving data to CSV...")
    save_data(X, y)

    # 3. 可视化数据分布
    print("\n[3/5] Plotting data distribution...")
    plot_data_distribution(X, y)

    # 4. 训练模型
    print("\n[4/5] Training Linear Regression model...")
    print("   Using Batch Gradient Descent")
    model = LinearRegression(learning_rate=0.02, n_iterations=1000)
    model.fit(X, y)

    # 5. 评估模型
    print("\n[5/5] Evaluating model...")
    r2_score = model.score(X, y)
    params = model.get_params()
    print(f"   Learned weights: {params['weights']}")
    print(f"   Learned bias: {params['bias']:.4f}")
    print(f"   R^2 Score: {r2_score:.4f}")

    # 6. 可视化训练过程与结果
    print("\n[Bonus] Generating visualizations...")
    plot_training_loss(model.loss_history)
    plot_regression_result(X, y, model, true_weights, true_bias)

    print("\n" + "=" * 50)
    print("Training completed! Check the 'figures/' directory.")
    print("=" * 50)

    return model
