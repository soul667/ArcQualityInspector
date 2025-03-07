import torch
from arc_quality_detection_pytorch import EnhancedArcQualityRNN

def main():
    # 创建简化版模型实例来展示结构
    input_size = 1
    hidden_size = 16
    num_layers = 1
    output_size = 1
    model = EnhancedArcQualityRNN(input_size, hidden_size, num_layers, output_size)
    
    # 打印模型结构
    print("\n简化模型结构:")
    print("-" * 50)
    print(model)
    
    # 计算并打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    lstm_params = 4 * hidden_size * (input_size + hidden_size + 1)  # 4 for LSTM gates
    fc_params = hidden_size * output_size + output_size  # weights + bias
    
    print("\n参数统计:")
    print("-" * 50)
    print(f"总参数量: {total_params:,}")
    print(f"LSTM参数量: {lstm_params:,}")
    print(f"全连接层参数量: {fc_params:,}")
    print("-" * 50)

if __name__ == "__main__":
    main()
