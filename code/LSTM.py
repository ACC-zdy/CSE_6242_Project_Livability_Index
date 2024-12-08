import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"数据加载成功，数据维度: {data.shape}")
    return data


# 归一化数据
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # 数据归一化到 [0, 1]
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# 准备数据：将时间序列转换为监督学习的格式
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])  # 取前time_step个数据作为输入特征
        y.append(data[i + time_step, 0])  # 取下一个数据作为目标值
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # 重新塑形以适应LSTM的输入格式
    return X, y


# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))  # 第一层LSTM
    model.add(Dropout(0.2))  # Dropout防止过拟合
    model.add(LSTM(units=50, return_sequences=False))  # 第二层LSTM
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # 输出层，回归问题预测一个值

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


# 训练模型
def train_lstm_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    model = build_lstm_model((X_train.shape[1], 1))  # 输入shape需要是 (时间步长, 特征数)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)
    return model, history


# 绘制训练过程中的损失曲线
def plot_loss(history):
    plt.plot(history.history['loss'], label='训练集损失')
    plt.plot(history.history['val_loss'], label='验证集损失')
    plt.title('LSTM 模型训练过程')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 模型评估和预测
def evaluate_and_predict(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)  # 逆归一化
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # 逆归一化

    # 计算模型的均方误差（MSE）
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"均方误差（MSE）：{mse}")

    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='真实值')
    plt.plot(y_pred, label='预测值')
    plt.title('LSTM 预测结果')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.legend()
    plt.show()

    return y_pred


# 主函数
def lstm_pipeline(file_path):
    # 1. 加载数据
    data = load_data(file_path)

    # 2. 归一化数据（假设数据的目标值在第一列）
    scaled_data, scaler = normalize_data(data.iloc[:, 0].values.reshape(-1, 1))  # 归一化第一列

    # 3. 创建训练数据集
    time_step = 60  # 选择时间步长为60，即用前60天的数据预测第61天的值
    X, y = create_dataset(scaled_data, time_step)

    # 4. 分割数据集：训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 5. 训练LSTM模型
    model, history = train_lstm_model(X_train, y_train, X_test, y_test)

    # 6. 绘制损失曲线
    plot_loss(history)

    # 7. 评估并预测
    y_pred = evaluate_and_predict(model, X_test, y_test, scaler)
    return model, y_pred


# 测试
if __name__ == "__main__":
    model, predictions = lstm_pipeline('data.csv')
