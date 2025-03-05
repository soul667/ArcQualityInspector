import torch
import numpy as np
import matplotlib.pyplot as plt
from arc_quality_detection_pytorch import EnhancedArcQualityRNN, normalize_sequence_length
from enhanced_arc_generator import ArcDataGenerator

def test_model(model_path, num_samples=6):
    """测试模型"""
    # 加载模型
    model = EnhancedArcQualityRNN(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    # 创建数据生成器
    generator = ArcDataGenerator(seq_length=500)
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    # 生成有缺陷和无缺陷的样本
    for i in range(num_samples):
        # 交替生成有缺陷和无缺陷样本
        force_defect = True if i % 2 == 0 else False
        sample = generator.generate_arc_sample(force_defect=force_defect)
        arc_data = sample['arc_points']
        true_label = sample['has_defect']
        defect_mask = sample['defect_mask']
        
        # Debug input data
        print(f"\nSample {i+1}:")
        print(f"True label: {'Defect' if force_defect else 'Normal'}")
        print(f"Generator label: {'Defect' if true_label else 'Normal'}")
        
        # Process and debug the data normalization steps
        normalized_length = normalize_sequence_length(arc_data)
        print(f"Sequence stats - Min: {normalized_length.min():.3f}, Max: {normalized_length.max():.3f}")
        
        normalized_data = (normalized_length - np.mean(normalized_length)) / np.std(normalized_length)
        print(f"Normalized stats - Min: {normalized_data.min():.3f}, Max: {normalized_data.max():.3f}")
        
        # Get model prediction with intermediate activations
        model_input = torch.FloatTensor(normalized_data).unsqueeze(0).unsqueeze(2)
        
        with torch.no_grad():
            # Forward pass with debugging
            prediction = model(model_input)
            prob = prediction.item()
            print(f"Model prediction: {prob:.6f}")
            
            # Check if defect regions exist
            if true_label == 1 and len(np.where(defect_mask == 1)[0]) > 0:
                defect_indices = np.where(defect_mask == 1)[0]
                print(f"Defect region: {defect_indices[0]} to {defect_indices[-1]}")
        
        # 绘制结果
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(normalized_data, label='Arc Data')
        
        # 如果有缺陷，显示缺陷区域
        if true_label == 1:
            defect_regions = np.where(defect_mask == 1)[0]
            if len(defect_regions) > 0:
                plt.axvspan(defect_regions[0], defect_regions[-1], color='red', alpha=0.2, label='Defect Region')
        
        plt.title(f'Sample {i+1} - True Label: {"Defect" if true_label else "Normal"}, '
                 f'Predicted Probability: {prob:.3f}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('arc_prediction.png')
    plt.close()
    
    print("预测结果已保存到 arc_prediction.png")

if __name__ == "__main__":
    model_path = 'best_arc_quality_model.pth'
    test_model(model_path)
