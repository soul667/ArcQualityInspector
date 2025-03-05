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

class EnhancedArcQualityRNN(nn.Module):
    """
    增强版圆弧质量检测RNN模型
    使用双向LSTM和注意力机制提高特征提取能力
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(EnhancedArcQualityRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.xavier_uniform_(self.attention_weights)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 激活函数和正则化
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def attention_net(self, lstm_output):
        """
        注意力机制计算
        
        参数:
            lstm_output: [batch_size, seq_len, hidden_size*2]
        返回:
            上下文向量: [batch_size, hidden_size*2]
        """
        attn_weights = torch.tanh(torch.matmul(lstm_output, self.attention_weights))  # [batch_size, seq_len, 1]
        soft_attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)  # [batch_size, hidden_size*2, 1]
        return context.squeeze(2)  # [batch_size, hidden_size*2]
    
    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)  # 双向*2
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch, seq_len, hidden*2]
        
        # 注意力机制
        attn_out = self.attention_net(lstm_out)  # [batch, hidden*2]
        
        # 批归一化
        out = self.batch_norm(attn_out)
        
        # 全连接层
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def normalize_sequence_length(sequence, target_length=500):
    """
    将序列标准化为指定长度
    
    参数:
        sequence: 输入序列
        target_length: 目标序列长度，默认500
        
    返回:
        标准化后的序列
    """
    current_length = len(sequence)
    if (current_length == target_length):
        return sequence
    
    # 创建等间隔的新索引点
    old_indices = np.arange(current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    
    # 使用线性插值调整序列长度
    return np.interp(new_indices, old_indices, sequence)

def normalize_mask_length(mask, target_length=500):
    """
    将缺陷掩码标准化为指定长度
    
    参数:
        mask: 输入掩码序列
        target_length: 目标序列长度，默认500
        
    返回:
        标准化后的掩码序列
    """
    current_length = len(mask)
    if (current_length == target_length):
        return mask
    
    # 创建等间隔的新索引点
    old_indices = np.arange(current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    
    # 使用最近邻插值保持二进制性质
    normalized_mask = np.interp(new_indices, old_indices, mask)
    return (normalized_mask > 0.5).astype(np.float32)  # 转换回二进制掩码

def load_cleaned_data():
    """
    从data目录加载处理过的圆弧数据，并将所有序列标准化为500点
    
    返回:
        X: 形状为 (num_samples, 500) 的特征数组
        y: 形状为 (num_samples,) 的标签数组，1表示有缺陷，0表示无缺陷
        defect_masks: 形状为 (num_samples, 500) 的掩码数组，标记缺陷位置
    """
    X = []
    y = []
    defect_masks = []
    
    # 加载所有cleaned数据
    for file in glob.glob('data/sample_*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            # 标准化序列长度为500点
            normalized_points = normalize_sequence_length(data['arc_points'])
            normalized_mask = normalize_mask_length(data['defect_mask'])
            X.append(normalized_points)
            y.append(data['label'])
            defect_masks.append(normalized_mask)
    
    if not X:
        raise ValueError("No data found in data directory")
    
    # 转换为numpy数组
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    defect_masks = np.array(defect_masks, dtype=np.float32)
    
    # 标准化数据
    X = (X - np.mean(X)) / np.std(X)
    
    return X, y, defect_masks

def print_data_stats(X, y, dataset_name="Dataset"):
    """
    Print dataset statistics
    """
    defect_count = np.sum(y == 1)
    no_defect_count = np.sum(y == 0)
    total_count = len(y)
    
    print(f"\n{dataset_name} Statistics:")
    print(f"Total samples: {total_count}")
    print(f"Defective samples: {defect_count} ({defect_count/total_count*100:.1f}%)")
    print(f"Non-defective samples: {no_defect_count} ({no_defect_count/total_count*100:.1f}%)")
    print(f"Sequence length: {X.shape[1]}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cpu'):
    """
    训练模型
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮次
        device: 设备 ('cpu' 或 'cuda')
    
    返回:
        history: 包含训练历史的字典
    """
    model.to(device)
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 100
    counter = 0
    
    print("\n开始训练:")
    print("-" * 60)
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
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
        
        # 更新学习率调度器并记录变化
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"\nEpoch {epoch+1}: Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
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
            torch.save(model.state_dict(), 'best_arc_quality_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"\n早停: {epoch+1} 轮次后验证损失未改善")
                break
    
    print("-" * 60)
    model.load_state_dict(torch.load('best_arc_quality_model.pth', weights_only=True))
    return history, model

def print_training_summary(history):
    """打印训练结果摘要"""
    best_epoch = np.argmin(history['val_loss'])
    print("\n训练结果:")
    print(f"最佳验证损失: {history['val_loss'][best_epoch]:.4f} (轮次 {best_epoch+1})")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终训练准确率: {history['train_acc'][-1]*100:.1f}%")
    print(f"最终验证准确率: {history['val_acc'][-1]*100:.1f}%")

def evaluate_model(model, test_loader, criterion, device='cpu'):
    """评估模型性能"""
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
    
    return test_loss, accuracy, cm, all_targets, all_predictions

def main():
    # 模型超参数
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2  # 增加到2层
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50000
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("正在加载数据...")
    X, y, _ = load_cleaned_data()
    print_data_stats(X, y)
    
    # Split datasets
    print("\nSplitting datasets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Print statistics for each set
    print_data_stats(X_train, y_train, "Training Set")
    print_data_stats(X_val, y_val, "Validation Set")
    print_data_stats(X_test, y_test, "Test Set")
    
    # 准备数据加载器
    print("\n准备数据加载器...")
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
    
    # 创建增强版模型
    input_size = 1
    output_size = 1
    model = EnhancedArcQualityRNN(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # 添加L2正则化
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    # 训练模型
    history, model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        scheduler,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    print_training_summary(history)
    
    # 评估模型
    test_loss, accuracy, cm, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\n测试集评估结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {accuracy*100:.1f}%")
    
    print("\n混淆矩阵:")
    print(cm)
    
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['无缺陷', '有缺陷']))

if __name__ == "__main__":
    main()
