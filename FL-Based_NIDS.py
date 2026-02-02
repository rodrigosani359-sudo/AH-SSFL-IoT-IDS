import glob
import numpy as np
from os import path
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# 强制开启 Eager Execution 以适配现代环境
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
import tensorflow as tf
import warnings
from dataPre import loadCsv, dataset_pre

# 强制开启 Eager Execution 和调试模式以适配现代 TensorFlow 环境
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

warnings.filterwarnings("ignore", category=Warning)

# --- 实验配置 ---
TL = 4
fl_epochs = 300

# --- 确保必要的目录存在 ---
if not os.path.exists("Server"):
    os.makedirs("Server")
if not os.path.exists("CentralServer"):
    os.makedirs("CentralServer")

# --- 数据加载与预处理 ---
trainPath_201 = "data/UNSW_NB15_Train201.csv"
trainPath_202 = "data/UNSW_NB15_Train202.csv"
trainPath_203 = "data/UNSW_NB15_Train203.csv"
trainPath_204 = "data/UNSW_NB15_Train204.csv"
trainPath_205 = "data/UNSW_NB15_Train205.csv"
testPath_2 = 'data/UNSW_NB15_TestBin.csv'

trainData_201 = loadCsv(trainPath_201)
trainData_202 = loadCsv(trainPath_202)
trainData_203 = loadCsv(trainPath_203)
trainData_204 = loadCsv(trainPath_204)
trainData_205 = loadCsv(trainPath_205)
testData_2 = loadCsv(testPath_2)

def get_x_y(data, TL):
    # 前196列为特征，最后一列为标签
    x_part = data[:, 0:196]
    y_part = data[:, 196]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_part)
    x_final = dataset_pre(x_scaled, TL)
    x_final = np.reshape(x_final, (-1, TL, 196))
    return x_final, y_part

x_train01, y_train01 = get_x_y(trainData_201, TL)
x_train02, y_train02 = get_x_y(trainData_202, TL)
x_train03, y_train03 = get_x_y(trainData_203, TL)
x_train04, y_train04 = get_x_y(trainData_204, TL)
x_train05, y_train05 = get_x_y(trainData_205, TL)
x_test, y_test = get_x_y(testData_2, TL)

shape = np.size(x_train01, axis=2)

# --- 模型定义与训练逻辑 ---
def get_compiled_model(shape):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(TL, shape)),
        keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
        keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def train_local_model(x_train, y_train, serverid, serverbs, serverepochs):
    model_path = "CentralServer/fl_model.h5"
    if path.exists(model_path):
        print(f"Loading Global Model for Client {serverid}...")
        # 强制重新编译以重置优化器状态，解决 "Unknown variable" 报错
        model = keras.models.load_model(model_path)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    else:
        model = get_compiled_model(shape)

    model.fit(x_train, y_train, batch_size=serverbs, epochs=serverepochs,
              validation_data=(x_test, y_test), verbose=2, shuffle=True)

    # 保存模型权重为 Object 数组以适配新版 NumPy
    weights = model.get_weights()
    np.save(f'Server/Server{serverid}', np.array(weights, dtype=object))
    return model

# --- 联邦聚合逻辑 ---
def load_models():
    arr = []
    # 按照 Server1, Server2... 的顺序加载，确保平均逻辑一致
    for i in range(1, 6):
        file_path = f"Server/Server{i}.npy"
        if path.exists(file_path):
            arr.append(np.load(file_path, allow_pickle=True))
    return np.array(arr, dtype=object)

def fl_average():
    arr = load_models()
    # 对所有客户端的权重矩阵进行平均聚合
    fl_avg = np.average(arr, axis=0)
    return fl_avg

def model_fl():
    avg_weights = fl_average()
    model = get_compiled_model(shape)
    model.set_weights(avg_weights)
    print("FL Global Model Ready!")
    
    # 评估全局模型
    print('Test Num:', len(y_test))
    score = model.evaluate(x_test, y_test, batch_size=20000, verbose=0)
    print('Global Model Score [Loss, Accuracy]:', score)
    
    # 保存为 H5 格式供下一轮迭代
    model.save("CentralServer/fl_model.h5")

# --- 主循环 ---
for i in range(fl_epochs):
    print(f"\n========== STARTING ROUND {i} ==========")
    train_local_model(x_train01, y_train01, 1, 500, 1)
    train_local_model(x_train02, y_train02, 2, 500, 1)
    train_local_model(x_train03, y_train03, 3, 500, 1)
    train_local_model(x_train04, y_train04, 4, 500, 1)
    train_local_model(x_train05, y_train05, 5, 500, 1)
    
    model_fl()
    
    # 清理内存防止 Colab 崩溃
    tf.keras.backend.clear_session()
    print(f"========== END OF ROUND {i} ==========\n")