import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import time

def generate_test_arc(seq_length=100, has_defect=None):
    """生成测试用的圆弧数据"""
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
    
    return arc_points, has_defect, defect_mask

def detect_arc_quality_onnx(session, arc_points):
    """使用ONNX模型检测圆弧质量"""
    # 准备输入数据 - 注意数据类型必须是float32
    input_data = arc_points.reshape(1, len(arc_points), 1).astype(np.float32)
    
    # 运行推理
    outputs = session.run(None, {'input': input_data})
    
    # 获取结果
    confidence = outputs[0][0][0]
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
    # 计时开始
    start_time = time.time()
    
    # 加载ONNX模型
    try:
        # 创建ONNX Runtime推理会话
        session = ort.InferenceSession("arc_quality_model.onnx")
        print("ONNX模型加载成功！")
    except Exception as e:
        print(f"ONNX模型加载失败: {e}")
        return
    
    # 打印模型加载时间
    load_time = time.time() - start_time
    print(f"模型加载耗时: {load_time:.2f}秒")
    
    # 生成并检测多个测试样例
    for i in range(5):
        # 生成测试样例
        arc_points, true_defect, defect_mask = generate_test_arc()
        
        # 检测质量（计时）
        inference_start = time.time()
        predicted_defect, confidence = detect_arc_quality_onnx(session, arc_points)
        inference_time = time.time() - inference_start
        
        # 打印结果
        print(f"\n测试样例 {i+1}:")
        print(f"实际状态: {'有缺陷' if true_defect == 1 else '无缺陷'}")
        print(f"预测结果: