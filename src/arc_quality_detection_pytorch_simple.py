import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
import glob
import json
import os

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

class SimpleArcQualityRNN(nn.Module):
    """
    简化版圆弧质量检测RNN模型
    使用单层单向LSTM，移除了复杂的注意力机制和多余的层
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(SimpleArcQualityRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 单向单层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # 简化的全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM层
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # Dropout和全连接层
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

def normalize_sequence_length(sequence, target_length=500):
    """将序列标准化为指定长度"""
    current_length = len(sequence)
    if (current_length == target_length):
        return sequence
    
    old_indices = np.arange(current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    return np.interp(new_indices, old_indices, sequence)

def load_cleaned_data():
    """从data目录加载处理过的圆弧数据"""
    X = []
    y = []
    
    for file in glob.glob('data/sample_*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            normalized_points = normalize_sequence_length(data['arc_points'])
            X.append(normalized_points)
            y.append(data['label'])
    
    if not X:
        raise ValueError("No data found in data directory")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # 标准化数据
    X = (X - np.mean(X)) / np.std(X)
    
    return X, y

def print_data_stats(X, y, dataset_name="Dataset"):
    """打印数据集统计信息"""
    defect_count = np.sum(y == 1)
    no_defect_count = np.sum(y == 0)
    total_count = len(y)
    
    print(f"\n{dataset_name} Statistics:")
    print(f"Total samples: {total_count}")
    print(f"Defective samples: {defect_count} ({defect_count/total_count*100:.1f}%)")
    print(f"Non-defective samples: {no_defect_count} ({no_defect_count/total_count*100:.1f}%)")
    print(f"Sequence length: {X.shape[1]}")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """训练模型"""
    model.to(device)
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 20  # 减少早停的耐心值，因为模型更简单了
    counter = 0
    
    print("\n开始训练:")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练模式
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{num_epochs} [{epoch_time:5.1f}s] "
              f"Train Loss: {train_loss:.4f} ({train_acc*100:5.1f}%) "
              f"Val Loss: {val_loss:.4f} ({val_acc*100:5.1f}%)")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_simple_arc_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"\n早停: {epoch+1} 轮次后验证损失未改善")
                break
    
    print("-" * 60)
    model.load_state_dict(torch.load('best_simple_arc_model.pth'))
    return history, model

def main():
    # 模型超参数
    BATCH_SIZE = 128
    HIDDEN_SIZE = 16  # 减小隐藏层大小
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("正在加载数据...")
    X, y = load_cleaned_data()
    print_data_stats(X, y)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 打印每个数据集的统计信息
    print_data_stats(X_train, y_train, "Training Set")
    print_data_stats(X_val, y_val, "Validation Set")
    print_data_stats(X_test, y_test, "Test Set")
    
    # 准备数据加载器
    X_train = torch.FloatTensor(X_train).unsqueeze(2)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val).unsqueeze(2)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(2)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 创建简化版模型
    input_size = 1
    output_size = 1
    model = SimpleArcQualityRNN(input_size, HIDDEN_SIZE, output_size)
    
    # 打印模型结构和参数量
    print("\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    history, model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    # 评估模型
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    
    print("\n开始测试评估...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
    cm = confusion_matrix(all_targets, all_predictions)
    
    print(f"\n测试集评估结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {accuracy*100:.1f}%")
    
    print("\n混淆矩阵:")
    print(cm)
    
    print("\n分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=['无缺陷', '有缺陷']))

if __name__ == "__main__":
    # 创建简化版模型实例来展示结构
    input_size = 1
    hidden_size = 16
    output_size = 1
    model = SimpleArcQualityRNN(input_size, hidden_size, output_size)
    
    # 打印模型结构
    print("\n简化模型结构:")
    print("-" * 50)
    print(model)
    
    # 计算并打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("\n参数统计:")
    print("-" * 50)
    print(f"总参数量: {total_params:,}")
    print(f"LSTM参数量: {4 * hidden_size * (input_size + hidden_size + 1):,}")  # 4 for LSTM gates
    print(f"全连接层参数量: {hidden_size * output_size + output_size:,}")  # weights + bias
