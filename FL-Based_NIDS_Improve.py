import os
import shutil
import numpy as np
import tensorflow as tf
from dataPre import loadCsv, dataset_pre
from google.colab import drive

# --- 1. 环境准备与云盘挂载 ---
print("正在执行 AH-SSFL 改进脚本...")
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

backup_path = "/content/drive/MyDrive/AH_SSFL_Project/Improved_Models/"
os.makedirs(backup_path, exist_ok=True)
os.makedirs('Server/', exist_ok=True) 
os.makedirs('CentralServer/', exist_ok=True)

# --- 2. 核心算法：自适应加权聚合 (AH-SSFL) ---
def fl_adaptive_weighted_average(client_accuracies):
    """
    权重分配公式: alpha_i = Acc_i / sum(Acc)
    """
    total_acc = sum(client_accuracies)
    if total_acc == 0:
        weights = [0.2] * 5
    else:
        weights = [acc / total_acc for acc in client_accuracies]
    
    print("\n" + "="*30)
    print("[AH-SSFL] Adaptive Weight Allocation:")
    for i, w in enumerate(weights):
        print(f" > Client {i+1}: Weight={w:.4f} (Acc={client_accuracies[i]:.4f})")
    print("="*30)
        
    path = 'Server/'
    client_models = [np.load(f'{path}Server{i+1}.npy', allow_pickle=True) for i in range(5)]
    
    new_weights = []
    for layer_idx in range(len(client_models[0])):
        weighted_layer = sum(client_models[i][layer_idx] * weights[i] for i in range(5))
        new_weights.append(weighted_layer)
    return new_weights

# --- 3. 改进版实验主循环 (101-200 轮) ---
def run_ah_ssfl_experiment(start_round=100, end_round=200, TL=4):
    model_path = 'CentralServer/fl_model.h5'
    global_model = tf.keras.models.load_model(model_path)
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    for r in range(start_round, end_round):
        print(f"\n[Round {r+1}] Federated Learning in progress...")
        client_accs = []
        
        for i in range(1, 6):
            data_file = f'data/UNSW_NB15_Train20{i}.csv'
            data = loadCsv(data_file)
            x_raw, y_raw = data[:, :196], data[:, 196]
            
            # 动态对齐修复
            x_processed = dataset_pre(x_raw, TL)
            actual_samples = x_processed.shape[0]
            y_final = y_raw[-actual_samples:] 
            x_final = np.reshape(x_processed, (actual_samples, TL, 196))
            
            # 本地微调
            history = global_model.fit(x_final, y_final, epochs=1, batch_size=512, validation_split=0.2, verbose=0)
            
            # NumPy  inhomogeneous shape 修复
            current_acc = history.history['val_acc'][0]
            client_accs.append(current_acc)
            weights_to_save = np.array(global_model.get_weights(), dtype=object) 
            np.save(f'Server/Server{i}.npy', weights_to_save, allow_pickle=True)
            
        # 自适应聚合
        improved_weights = fl_adaptive_weighted_average(client_accs)
        global_model.set_weights(improved_weights)
        
        # 实时保存与云端同步
        save_name = f'ahssfl_model_r{r+1}.h5'
        global_model.save(f'CentralServer/{save_name}')
        shutil.copy(f'CentralServer/{save_name}', backup_path + save_name)
        print(f"✅ Round {r+1} completed and backed up to Drive.")

if __name__ == "__main__":
    run_ah_ssfl_experiment()
