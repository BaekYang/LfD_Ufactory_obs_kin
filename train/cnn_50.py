import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# CNN 학습 관련 파라미터 설정
class paramModel():
    num_filters = 128      # CNN 필터 수
    kernel_size = 3       # 컨볼루션 커널 사이즈
    lr = 1e-4
    epochs = 50000
    batch_size = 8
    weight_decay = 1e-4
    log_dir = r"./created_model/logs/cnn_model"

pm = paramModel()

# Dataset 정의 (기존과 동일)
class XArm4SecDataset(torch.utils.data.Dataset):
    def __init__(self, data_path="../data/pumping result"):
        super().__init__()
        self.data_path = data_path
        self.csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        if not self.csv_files:
            raise FileNotFoundError(f"[ERROR] No CSV files found in {data_path}")
        
        self.samples = []
        for fpath in self.csv_files:
            seq_data = self._load_and_filter_csv(fpath)
            if seq_data.shape[0] < 2:
                continue
            input_seq = seq_data[:-1]
            target_seq = seq_data[1:]
            self.samples.append((input_seq, target_seq))
    
    def _load_and_filter_csv(self, fpath):
        filtered_rows = []
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # timestamp 및 좌표, 각도 값들은 빈 문자열인 경우 0.0으로 대체
                    t_ms = float(row['timestamp']) if row['timestamp'] != '' else 0.0
                    if t_ms > 10000.0:
                        break  # 10초까지만 사용

                    x = float(row['x']) if row['x'] != '' else 0.0
                    y = float(row['y']) if row['y'] != '' else 0.0
                    z = float(row['z']) if row['z'] != '' else 0.0
                    roll = float(row['r']) if row['r'] != '' else 0.0
                    pitch = float(row['p']) if row['p'] != '' else 0.0
                    yaw = float(row['y']) if row['y'] != '' else 0.0

                    # joint 값들도 빈 문자열일 경우 기본값 0.0으로 처리
                    j1 = float(row['joint1']) if row['joint1'] != '' else 0.0
                    j2 = float(row['joint2']) if row['joint2'] != '' else 0.0
                    j3 = float(row['joint3']) if row['joint3'] != '' else 0.0
                    j4 = float(row['joint4']) if row['joint4'] != '' else 0.0
                    j5 = float(row['joint5']) if row['joint5'] != '' else 0.0
                    j6 = float(row['joint6']) if row['joint6'] != '' else 0.0
                except Exception as e:
                    print(f"CSV 파싱 오류: {row} - {e}")
                    continue

                filtered_rows.append([x, y, z, roll, pitch, yaw, j1, j2, j3, j4, j5, j6])
        if len(filtered_rows) == 0:
            return np.zeros((0, 12), dtype=np.float32)
        arr = np.array(filtered_rows, dtype=np.float32)
        return arr

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Custom collate 함수 (시퀀스 길이를 맞추기 위해 0 패딩)
def custom_collate_fn(batch):
    max_len = max(sample[0].shape[0] for sample in batch)
    batch_input_seqs = []
    batch_target_seqs = []
    
    for input_seq, target_seq in batch:
        cur_len = input_seq.shape[0]
        pad_len = max_len - cur_len
        padded_input = np.pad(input_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        padded_target = np.pad(target_seq, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        batch_input_seqs.append(torch.tensor(padded_input, dtype=torch.float))
        batch_target_seqs.append(torch.tensor(padded_target, dtype=torch.float))
    
    # 결과 shape: [seq_len, batch, feature]
    batch_input_seqs = torch.stack(batch_input_seqs, dim=0).transpose(0, 1)
    batch_target_seqs = torch.stack(batch_target_seqs, dim=0).transpose(0, 1)
    
    return batch_input_seqs, batch_target_seqs

# SimpleCNN 모델: 1D Conv를 사용하여 시퀀스 예측 (padding 수정)
class SimpleCNN(nn.Module):
    def __init__(self, input_size=12, num_filters=pm.num_filters, kernel_size=pm.kernel_size, output_size=12):
        super().__init__()
        # "same" convolution을 위해 padding을 (kernel_size-1)//2 로 설정 (kernel_size가 3이면 padding=1)
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, 
                               kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                               kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=num_filters, out_channels=output_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, feature, seq_len]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        # out: [batch, output_size, seq_len] (입력과 동일한 시퀀스 길이)
        return out

# ===== 추가된 RMSE, R² Score 계산 함수 =====
def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0
# ========================================

# 전체 학습 및 추론 코드 (CNN 기반)
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # 데이터셋 및 DataLoader 생성
    dataset = XArm4SecDataset()  # 기본값 사용
    print(f"[INFO] Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        print("[WARN] No valid data.")
        return
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=pm.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    # 모델, 옵티마이저, 손실 함수 정의
    model = SimpleCNN(input_size=12, num_filters=pm.num_filters, kernel_size=pm.kernel_size, output_size=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=pm.lr, weight_decay=pm.weight_decay)
    criterion = nn.MSELoss()
    
    # TensorBoard Writer 설정
    writer = SummaryWriter(log_dir=pm.log_dir)
    
    # TensorBoard: 모델 그래프 로그 추가 (추가된 부분)
    dummy_input = torch.zeros((pm.batch_size, 12, 100), device=device)  # 임의의 입력 (시퀀스 길이: 100)
    writer.add_graph(model, dummy_input)
    
    print("[INFO] Start Training with CNN...")
    for epoch in tqdm(range(pm.epochs), desc="Training Progress"):
        model.train()
        total_loss = 0.0
        
        for input_seq, target_seq in loader:
            # input_seq, target_seq: [seq_len, batch, feature] → [batch, feature, seq_len]
            input_seq = input_seq.permute(1, 2, 0).to(device)
            target_seq = target_seq.permute(1, 2, 0).to(device)
            
            optimizer.zero_grad()
            pred = model(input_seq)  # [batch, output_size, seq_len]
            loss = criterion(pred, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{pm.epochs}] - Loss: {avg_loss:.6f}")
    
    print("[INFO] Training Done.\n")
    
    # 모델 저장
    output_dir = r"./created_model"
    os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성
    save_model_path = os.path.join(output_dir, "cnn_model.pt")
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f"[INFO] Model saved at: {save_model_path}")
    
    writer.close()
    
    # --- 추론 (Inference) ---
    model.eval()
    sample_input, _ = next(iter(loader))
    # sample_input: [seq_len, batch, feature] → [batch, feature, seq_len]
    sample_input_cnn = sample_input.permute(1, 2, 0).to(device)
    with torch.no_grad():
        pred = model(sample_input_cnn)
    # 예측 결과를 다시 [seq_len, batch, feature]로 변환
    pred = pred.permute(2, 0, 1).cpu().numpy()
    
    # 추론 결과 CSV 저장 (첫 배치의 첫 샘플 사용)
    save_csv = os.path.join(output_dir, "inference_result.csv")
    with open(save_csv, 'w', newline='') as f:
        fieldnames = [
            'timestamp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        sample_pred = pred[:, 0, :]  # [seq_len, feature]
        for i, vec in enumerate(sample_pred):
            row = {
                'timestamp': i,
                'x': vec[0],
                'y': vec[1],
                'z': vec[2],
                'roll': vec[3],
                'pitch': vec[4],
                'yaw': vec[5],
                'joint1': vec[6],
                'joint2': vec[7],
                'joint3': vec[8],
                'joint4': vec[9],
                'joint5': vec[10],
                'joint6': vec[11],
            }
            writer_csv.writerow(row)
    print(f"[INFO] Inference CSV saved => {save_csv}")
    
    # --- RMSE, R² Score 평가 블록 (추가된 부분) ---
    sample_input_metric, sample_target_metric = next(iter(loader))
    sample_input_cnn_metric = sample_input_metric.permute(1, 2, 0).to(device)
    with torch.no_grad():
        pred_metric = model(sample_input_cnn_metric)
    pred_metric = pred_metric.permute(2, 0, 1).cpu().numpy()
    sample_target_metric = sample_target_metric.numpy()
    
    # 전체 배치의 데이터를 1차원으로 평탄화하여 계산
    rmse_value = compute_rmse(sample_target_metric.flatten(), pred_metric.flatten())
    r2_value = compute_r2_score(sample_target_metric.flatten(), pred_metric.flatten())
    
    print(f"[INFO] RMSE on sample batch: {rmse_value:.6f}")
    print(f"[INFO] R² Score on sample batch: {r2_value:.6f}")
    # ============================================
    
    # --- 시각화 ---
    print("[INFO] Start Visualization...")
    t_axis = list(range(sample_pred.shape[0]))
    x_vals = sample_pred[:, 0]
    y_vals = sample_pred[:, 1]
    z_vals = sample_pred[:, 2]
    
    # 3D 궤적 그래프
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Trajectory (x,y,z)")
    plt.show()
    
    # 2D 그래프 (Roll, Pitch, Yaw)
    roll_vals = sample_pred[:, 3]
    pitch_vals = sample_pred[:, 4]
    yaw_vals = sample_pred[:, 5]
    plt.figure()
    plt.plot(t_axis, roll_vals, label='roll')
    plt.plot(t_axis, pitch_vals, label='pitch')
    plt.plot(t_axis, yaw_vals, label='yaw')
    plt.xlabel('Time index')
    plt.ylabel('Angle')
    plt.title("Roll/Pitch/Yaw over time")
    plt.legend()
    plt.show()
    
    # Joint angles 2D 그래프
    joint_vals = sample_pred[:, 6:]
    plt.figure()
    for i in range(6):
        plt.plot(t_axis, joint_vals[:, i], label=f"joint{i+1}")
    plt.xlabel('Time index')
    plt.ylabel('Joint angle')
    plt.title("Joint angles over time")
    plt.legend()
    plt.show()
    
    print("[INFO] Visualization Done.")

if __name__ == "__main__":
    main()
