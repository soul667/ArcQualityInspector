import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from arc_quality_detection_pytorch import (
    EnhancedArcQualityRNN,
    normalize_sequence_length,
    normalize_mask_length,
    load_cleaned_data
)
from enhanced_arc_generator import ArcDataGenerator

def load_model(model_path='best_arc_quality_model.pth'):
    """加载训练好的模型并打印参数统计信息"""
    # 设置模型参数
    input_size = 1
    hidden_size = 24  # 匹配新模型的隐藏层大小
    num_layers = 2    # 使用2层LSTM
    output_size = 1
    
    # 创建模型实例
    model = EnhancedArcQualityRNN(input_size, hidden_size, num_layers, output_size)
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("\n模型参数统计:")
    print("-" * 50)
    print(f"总参数量: {total_params:,}")
    print("-" * 50)
    
    return model

def test_random_samples(num_samples=10):
    """随机测试数据样本"""
    print("\n开始随机测试样本...")
    print("-" * 50)
    
    # 生成一半样本
    generator = ArcDataGenerator(seq_length=500)
    gen_X, gen_y, gen_masks, _ = generator.generate_dataset(num_samples // 2)
    
    # 加载另一半数据
    load_X, load_y, load_masks = load_cleaned_data()
    total_load_samples = len(load_X)
    load_indices = random.sample(range(total_load_samples), num_samples // 2)
    
    # 合并数据
    X = np.concatenate([gen_X, load_X[load_indices]])
    y = np.concatenate([gen_y, load_y[load_indices]])
    
    # 加载模型
    model = load_model()
    
    # 测试每个样本
    for i in range(len(X)):
        # 准备输入数据
        input_data = torch.FloatTensor(X[i]).unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
        
        # 获取预测结果
        with torch.no_grad():
            output = model(input_data)
            predicted = (output > 0.5).float().item()
        
        # 获取真实标签
        true_label = y[i]
        
        # 打印结果
        print(f"\n样本 #{i} ({'生成' if i < num_samples//2 else '加载'}):")
        print(f"真实标签: {'有缺陷' if true_label == 1 else '无缺陷'}")
        print(f"预测结果: {'有缺陷' if predicted == 1 else '无缺陷'}")
        print(f"预测概率: {output.item():.4f}")
        
        # 判断预测是否正确
        if predicted == true_label:
            print("预测正确 ✓")
        else:
            print("预测错误 ✗")
        print("-" * 50)
        
        # 绘制波形图
        plt.figure(figsize=(12, 4))
        plt.plot(X[i], label='Arc Points')
        plt.title(f'Sample #{i} ({("Generated" if i < num_samples//2 else "Loaded")}) - Ground Truth: {"Defective" if true_label == 1 else "Non-Defective"}\nPrediction: {"Defective" if predicted == 1 else "Non-Defective"} (Probability: {output.item():.4f})')
        plt.xlabel('Point Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 测试随机样本
    test_random_samples()
