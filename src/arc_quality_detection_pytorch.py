import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

class ArcQualityRNN(nn.Module):
    """
    用于检测圆弧质量的RNN模型
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(ArcQualityRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = out[:, -1, :]
        
        # 批归一化
        out = self.batch_norm(out)
        
        # 全连接层
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

def generate_arc_data(num_samples, seq_length, radius=1.0):
    """
    生成圆弧数据，其中随机区间质量不好
    
    参数:
        num_samples: 样本数量
        seq_length: 序列长度（每个圆弧上的点数）
        radius: 圆弧半径
    
    返回:
        X: 形状为 (num_samples, seq_length) 的特征数组
        y: 形状为 (num_samples,) 的标签数组，1表示有缺陷，0表示无缺陷
        defect_masks: 形状为 (num_samples, seq_length) 的掩码数组，标记缺陷位置
    """
    X = []
    y = []
    defect_masks = []
    
    for i in range(num_samples):
        # 生成等间隔角度
        angles = np.linspace(0, np.pi/2, seq_length)  # 生成90度的圆弧
        
        # 生成完美圆弧
        perfect_arc = radius * np.sin(angles)
        
        # 添加少量基础噪声
        base_noise = np.random.normal(0, 0.01, seq_length)
        arc_points = perfect_arc + base_noise
        
        # 决定是否有缺陷
        has_defect = np.random.choice([0, 1])
        
        # 创建缺陷掩码，全0初始化
        defect_mask = np.zeros(seq_length)
        
        if has_defect:
            # 确定缺陷区间长度（序列长度的10%到30%）
            defect_length = np.random.randint(seq_length // 10, seq_length // 3)
            
            # 确定缺陷起始位置
            defect_start = np.random.randint(0, seq_length - defect_length)
            defect_end = defect_start + defect_length
            
            # 更新缺陷掩码
            defect_mask[defect_start:defect_end] = 1
            
            # 随机选择缺陷类型
            defect_type = np.random.randint(0, 4)
            
            if defect_type == 0:  # 噪声增大
                noise_level = np.random.uniform(0.1, 0.3)
                arc_points[defect_start:defect_end] += np.random.normal(0, noise_level, defect_length)
            
            elif defect_type == 1:  # 局部变形
                deformation = np.random.normal(0, 0.2, defect_length)
                arc_points[defect_start:defect_end] += deformation
            
            elif defect_type == 2:  # 局部半径变化
                radius_change = np.random.uniform(-0.3, 0.3)
                original_values = perfect_arc[defect_start:defect_end]
                arc_points[defect_start:defect_end] = original_values * (1 + radius_change)
            
            else:  # 高频波纹
                freq_factor = np.random.randint(3, 8)
                local_angles = np.linspace(0, np.pi * freq_factor, defect_length)
                wave_amplitude = np.random.uniform(0.05, 0.15)
                arc_points[defect_start:defect_end] += wave_amplitude * np.sin(local_angles)
        
        X.append(arc_points)
        y.append(has_defect)
        defect_masks.append(defect_mask)
    
    return np.array(X), np.array(y), np.array(defect_masks)

def visualize_samples(X, y, defect_masks, num_samples=5):
    """
    可视化一些样本数据
    """
    plt.figure(figsize=(15, 10))
    
    # 计数器，确保我们能找到足够的样本
    defect_count = 0
    no_defect_count = 0
    
    for i in range(len(X)):
        if y[i] == 1 and defect_count < num_samples:
            plt.subplot(2, num_samples, defect_count + 1)
            plt.plot(X[i])
            
            # 高亮缺陷区域
            defect_indices = np.where(defect_masks[i] == 1)[0]
            if len(defect_indices) > 0:
                plt.axvspan(defect_indices[0], defect_indices[-1], alpha=0.3, color='red')
                
            plt.title(f"有缺陷圆弧 {defect_count+1}")
            defect_count += 1
        
        elif y[i] == 0 and no_defect_count < num_samples:
            plt.subplot(2, num_samples, no_defect_count + 1 + num_samples)
            plt.plot(X[i])
            plt.title(f"无缺陷圆弧 {no_defect_count+1}")
            no_defect_count += 1
        
        # 如果两种类型的样本都已经有足够数量，则退出循环
        if defect_count >= num_samples and no_defect_count >= num_samples:
            break
    
    plt.tight_layout()
    plt.savefig('arc_samples.png')
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """
    训练模型
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
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
    patience = 10
    counter = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练模式
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
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
                
                # 计算准确率
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
        print(f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_arc_quality_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_arc_quality_model.pth'))
    return history, model

def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_loader, criterion, device='cpu'):
    """评估模型性能"""
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            # 获取预测
            predicted = (outputs > 0.5).float()
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    
    # 计算准确率
    accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    return test_loss, accuracy, cm, all_targets, all_predictions

def main():
    # 设置参数
    NUM_SAMPLES = 2000
    SEQ_LENGTH = 100
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成数据
    X, y, defect_masks = generate_arc_data(NUM_SAMPLES, SEQ_LENGTH)
    
    # 可视化部分样本
    visualize_samples(X, y, defect_masks)
    
    # 分割数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).unsqueeze(2)  # [batch, seq_len, feature_dim]
    y_train = torch.FloatTensor(y_train).unsqueeze(1)  # [batch, 1]
    
    X_val = torch.FloatTensor(X_val).unsqueeze(2)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    
    X_test = torch.FloatTensor(X_test).unsqueeze(2)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 创建模型
    input_size = 1  # 一维特征
    output_size = 1  # 二分类
    model = ArcQualityRNN(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size
    )
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 打印模型结构
    print(model)
    
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
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    test_loss, accuracy, cm, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\n测试集上的损失: {test_loss:.4f}")
    print(f"测试集上的准确率: {accuracy:.4f}")
    
    print("\n混淆矩阵:")
    print(cm)
    
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['无缺陷', '有缺陷']))

if __name__ == "__main__":
    main()