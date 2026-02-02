import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def loadCsv(loadPath):
    """
    针对 Colab 优化的数据加载函数
    1. 自动处理字符串特征 (Label Encoding)
    2. 自动归一化 (Min-Max Scaling)
    3. 限制读取行数以防止内存溢出 (OOM)
    """
    print(f"--- 正在加载并预处理数据: {loadPath} ---")
    
    # 为了防止 Colab 内存崩溃，默认读取前 100,000 行
    # 如果你的内存充足，可以删除 nrows 参数或将其改大
    try:
        df = pd.read_csv(loadPath, header=None, nrows=100000, low_memory=False)
    except FileNotFoundError:
        print(f"错误：找不到文件 {loadPath}，请检查路径和文件名是否正确。")
        return None

    # 1. 处理分类特征 (将字符串转为数字)
    # 原始 UNSW-NB15 数据集包含 proto, service, state 等字符型特征
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    # 2. 缺失值处理
    df = df.fillna(0)
    
    # 3. 归一化处理 (神经网络训练必需步骤)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    
    print(f"加载完成！数据形状为: {data_scaled.shape}")
    return data_scaled

def dataset_pre(data, TL):
    """
    时间序列特征构造函数
    data: 输入的二维数据
    TL: 时间窗口跨度 (Time Length)
    """
    if data is None:
        return None
        
    # 如果时间跨度为 1，直接返回原数据，节省计算资源
    if TL <= 1:
        return data

    Data_2 = []
    size = np.size(data, axis=0)
    
    # 优化后的平滑处理逻辑
    for k in range(0, size):
        # 构造滑动窗口特征
        if k + TL <= size:
            window = data[k : k + TL].flatten()
        else:
            # 末尾填充逻辑：保持与原作者逻辑一致
            last_rows = data[k : size]
            padding_needed = TL - len(last_rows)
            padding = np.tile(data[size-1], (padding_needed, 1))
            window = np.vstack([last_rows, padding]).flatten()
        
        Data_2.append(window)
        
    Features = np.array(Data_2)
    return Features

if __name__ == "__main__":
    # 此处仅用于本地测试
    print("dataPre 模块已就绪。")