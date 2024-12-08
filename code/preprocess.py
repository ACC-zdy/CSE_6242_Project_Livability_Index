import pandas as pd
import numpy as np

# 加载数据
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"数据加载成功，数据维度: {data.shape}")
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

# 处理缺失值
def handle_missing_values(data):
    # 可以选择填充缺失值、删除含有缺失值的行或列
    # 填充缺失值为均值/中位数/众数
    data.fillna(data.mean(), inplace=True)  # 用均值填充
    print("缺失值处理完成。")
    return data

# 数据去重
def remove_duplicates(data):
    # 去重
    data.drop_duplicates(inplace=True)
    print("重复数据去重完成。")
    return data

# 特征工程 - 编码分类变量
def encode_categorical_features(data):
    # 假设我们有一个分类变量需要进行标签编码
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
        print(f"列 {col} 编码完成。")
    return data

# 归一化/标准化
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def normalize_features(data, method='standardize'):
    # 标准化（均值为0，方差为1）
    if method == 'standardize':
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    # 归一化（将数据缩放至[0, 1]）
    elif method == 'normalize':
        scaler = MinMaxScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    print(f"数据 {method} 完成。")
    return data

# 数据预处理主函数
def preprocess_data(file_path):
    data = load_data(file_path)
    if data is None:
        return None

    data = handle_missing_values(data)
    data = remove_duplicates(data)
    data = encode_categorical_features(data)
    data = normalize_features(data, method='standardize')  # 可选择 'normalize'

    return data

# 测试
if __name__ == "__main__":
    processed_data = preprocess_data('data.csv')
    if processed_data is not None:
        processed_data.to_csv('processed_data.csv', index=False)
        print("预处理后的数据已保存为 'processed_data.csv'")
