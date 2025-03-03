import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_arc_quality_model(model_path='best_arc_quality_model.h5'):
    """加载训练好的模型"""
    return load_model(model_path)

def detect_arc_quality(model, arc_points):
    """
    检测单个圆弧的质量
    :param model: 训练好的模型
    :param arc_points: 形状为 (seq_length,) 的一维数组
    :return: 质量预测 (0=低质量, 1=高质量)
    """
    # 重塑输入为模型所需的形状
    X = arc_points.reshape(1, len(arc_points), 1)
    
    # 进行预测
    prediction = model.predict(X)[0][0]
    
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

def generate_test_arc(seq_length=100, quality=None):
    """
    生成测试用的圆弧
    :param seq_length: 序列长度
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
    except:
        print("模型加载失败，请确保已经训练并保存了模型。")
        return
    
    # 生成一些测试圆弧并进行预测
    for i in range(5):
        # 生成测试圆弧
        arc_points, true_quality = generate_test_arc()
        
        # 检测质量
        predicted_quality, probability = detect_arc_quality(model, arc_points)
        
        # 打印结果
        print(f"测试圆弧 {i+1}:")
        print(f"真实质量: {'高质量' if true_quality == 1 else '低质量'}")
        print(f"预测质量: {'高质量' if predicted_quality == 1 else '低质量'}")
        print(f"预测概率: {probability:.4f}")
        print("-" * 50)
        
        # 可视化
        visualize_prediction(arc_points, predicted_quality, probability)

if __name__ == "__main__":
    main()