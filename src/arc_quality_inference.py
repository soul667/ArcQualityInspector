import torch
import numpy as np
import matplotlib.pyplot as plt
from arc_quality_detection_pytorch import ArcQualityRNN

def load_model(model_path='best_arc_quality_model.pth'):
    """加载训练好的模型"""
    # 创建模型结构
    model = ArcQualityRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_test_arc(seq_length=100, has_defect=None, defect_position=None):
    """
    生成测试用的圆弧
    
    参数:
        seq_length: 序列长度
        has_defect: 如果为None，随机选择；否则为0（无缺陷）或1（有缺陷）
        defect_position: 如果None且has_defect=1，随机选择缺陷位置；
                        否则为 (start, end) 元组
    
    返回:
        arc_points: 圆弧点序列
        has_defect: 是否有缺陷 (0/1)
        defect_mask: 缺陷位置掩码
    """
    # 生成等间隔角度
    angles = np.linspace(0, np.pi/2, seq_length)
    radius = 1.0
    
    # 生成完美圆弧
    perfect_arc = radius * np.sin(angles)
    
    # 添加少量基础噪声
    base_noise = np.random.normal(0, 0.01, seq_length)
    arc_points = perfect_arc + base_noise
    
    # 确定是否有缺陷
    if has_defect is None:
        has_defect = np.random.choice([0, 1])
    
    # 初始化缺陷掩码
    defect_mask = np.zeros(seq_length)
    
    if has_defect:
        # 确定缺陷区间
        if defect_position is None:
            defect_length = np.random.randint(seq_length // 10, seq_length // 3)
            defect_start = np.random.randint(0, seq_length - defect_length)
            defect_end = defect_start + defect_length
        else:
            defect_start, defect_end = defect_position
            defect_length = defect_end - defect_start
        
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
    
    return arc_points, has_defect, defect_mask

def detect_arc_quality(model, arc_points):
    """
    检测圆弧质量
    
    参数:
        model: 训练好的模型
        arc_points: 形状为 (seq_length,) 的一维数组
    
    返回:
        has_defect: 预测是否有缺陷 (0/1)
        confidence: 预测的置信度 [0,1]
    """
    # 转换为PyTorch张量
    x = torch.FloatTensor(arc_points).unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
    
    # 进行预测
    with torch.no_grad():
        output = model(x)
    
    # 获取预测结果
    confidence = output.item()
    has_defect = 1 if confidence >= 0.5 else 0
    
    return has_defect, confidence

def visualize_detection(arc_points, true_defect, predicted_defect, confidence, defect_mask=None):
    """可视化检测结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(arc_points, color='blue', label='圆弧')
    
    # 如果有缺陷掩码，标记真实缺陷区域
    if defect_mask is not None and true_defect == 1:
        defect_indices = np.where(defect_mask == 1)[0]
        if len(defect_indices) > 0:
            plt.axvspan(defect_indices[0], defect_indices[-1], alpha=0.3, color='red', label='实际缺陷区域')
    
    # 获取预测结果文本
    prediction_text = "有缺陷" if predicted_defect == 1 else "无缺陷"
    truth_text = "有缺陷" if true_defect == 1 else "无缺陷"
    
    # 设置标题和标签
    plt.title(f"圆弧质量检测\n预测: {prediction_text} (置信度: {confidence:.4f}), 实际: {truth_text}")
    plt.xlabel("点索引")
    plt.ylabel("值")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载模型
    try:
        model = load_model()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 生成并检测多个测试样例
    for i in range(5):
        # 生成测试样例
        arc_points, true_defect, defect_mask = generate_test_arc()
        
        # 检测质量
        predicted_defect, confidence = detect_arc_quality(model, arc_points)
        
        # 打印结果
        print(f"\n测试样例 {i+1}:")
        print(f"实际状态: {'有缺陷' if true_defect == 1 else '无缺陷'}")
        print(f"预测结果: {'有缺陷' if predicted_defect == 1 else '无缺陷'}")
        print(f"预测置信度: {confidence:.4f}")
        
        # 可视化结果
        visualize_detection(arc_points, true_defect, predicted_defect, confidence, defect_mask)

if __name__ == "__main__":
    main()