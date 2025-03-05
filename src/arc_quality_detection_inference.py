import numpy as np
import matplotlib.pyplot as plt
import torch
from arc_quality_detection_pytorch import EnhancedArcQualityRNN

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
    if current_length == target_length:
        return sequence
    
    # 创建等间隔的新索引点
    old_indices = np.arange(current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)
    
    # 使用线性插值调整序列长度
    return np.interp(new_indices, old_indices, sequence)

def load_arc_quality_model(model_path='best_arc_quality_model.pth'):
    """加载训练好的模型"""
    model = EnhancedArcQualityRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def detect_arc_quality(model, arc_points):
    """
    检测单个圆弧的质量
    :param model: 训练好的模型
    :param arc_points: 任意长度的一维数组
    :return: 质量预测 (0=低质量, 1=高质量)
    """
    # 标准化序列长度为500点
    normalized_points = normalize_sequence_length(arc_points)
    
    # 重塑输入为模型所需的形状
    X = torch.FloatTensor(normalized_points).reshape(1, 500, 1)
    
    # 进行预测
    with torch.no_grad():
        prediction = model(X).item()
    
    # 将预测概率转换为类别
    quality_class = 1 if prediction >= 0.5 else 0
    
    return quality_class, prediction

def visualize_prediction(arc_points, quality_class, probability):
    """可视化圆弧和预测结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(arc_points)
    quality_text = "高质量" if quality_class == 1 else "低质量"
    plt.title(f"圆弧质量检测: {quality_text} (概率: {probability:.4f})")
    plt.xlabel("点索引")
    plt.ylabel("值")
    plt.grid(True)
    plt.savefig('arc_prediction.png')
    plt.show()

def generate_test_arc(seq_length=500, quality=None):
    """
    生成测试用的圆弧
    :param seq_length: 序列长度，默认500
    :param quality: 如果为None，随机生成；否则为0（低质量）或1（高质量）
    :return: 圆弧点序列和真实质量标签
    """
    angles = np.linspace(0, np.pi/2, seq_length)
    radius = 1.0
    
    if quality is None:
        quality = np.random.randint(0, 2)
    
    if quality == 1:  # 高质量圆弧
        noise_level = np.random.uniform(0.01, 0.05)
        arc_points = radius * np.sin(angles) + np.random.normal(0, noise_level, seq_length)
    else:  # 低质量圆弧
        defect_type = np.random.randint(0, 4)
        if defect_type == 0:  # 较大噪声
            noise_level = np.random.uniform(0.1, 0.3)
            arc_points = radius * np.sin(angles) + np.random.normal(0, noise_level, seq_length)
        elif defect_type == 1:  # 局部变形
            arc_points = radius * np.sin(angles)
            defect_start = np.random.randint(0, seq_length - seq_length//5)
            defect_length = seq_length // 5
            arc_points[defect_start:defect_start+defect_length] += np.random.normal(0, 0.2, defect_length)
        elif defect_type == 2:  # 半径变化
            varying_radius = radius + np.random.normal(0, 0.1, seq_length)
            arc_points = varying_radius * np.sin(angles)
        else:  # 非圆弧形状
            arc_points = radius * np.sin(angles) + 0.2 * np.sin(3 * angles)
    
    return arc_points, quality

def main():
    # 加载模型
    try:
        model = load_arc_quality_model()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        print("请确保已经训练并保存了模型。")
        return
    
    # 生成不同长度的测试圆弧并进行预测
    test_lengths = [40, 100, 200, 500]  # 测试不同长度的输入
    
    for length in test_lengths:
        print(f"\n测试长度为 {length} 的圆弧:")
        # 生成测试圆弧
        arc_points, true_quality = generate_test_arc(seq_length=length)
        
        # 检测质量
        predicted_quality, probability = detect_arc_quality(model, arc_points)
        
        # 打印结果
        print(f"输入长度: {length}")
        print(f"真实质量: {'高质量' if true_quality == 1 else '低质量'}")
        print(f"预测质量: {'高质量' if predicted_quality == 1 else '低质量'}")
        print(f"预测概率: {probability:.4f}")
        print("-" * 50)
        
        # 可视化
        visualize_prediction(arc_points, predicted_quality, probability)

if __name__ == "__main__":
    main()
