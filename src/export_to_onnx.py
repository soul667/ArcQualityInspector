import torch
import numpy as np
import argparse
from arc_quality_detection_pytorch import EnhancedArcQualityRNN

def export_model_to_onnx(model_path, onnx_path, seq_length=500):
    """将PyTorch模型导出为ONNX格式"""
    # 加载训练好的模型
    model = EnhancedArcQualityRNN(input_size=1, hidden_size=64, num_layers=1, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    print(f"加载PyTorch模型: {model_path}")
    
    # 创建示例输入
    dummy_input = torch.randn(1, seq_length, 1)
    
    # 导出到ONNX格式
    torch.onnx.export(
        model,                    # 要导出的模型
        dummy_input,              # 模型输入示例
        onnx_path,                # 保存路径
        export_params=True,       # 存储训练参数
        opset_version=12,         # ONNX操作集版本
        do_constant_folding=True, # 优化常量折叠
        input_names=['input'],    # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},   # 动态批大小
            'output': {0: 'batch_size'}
        }
    )
    print(f"模型已成功导出到 {onnx_path}")
    
    # 验证模型
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path)
    
    # 创建测试输入
    test_input = np.random.randn(1, seq_length, 1).astype(np.float32)
    
    # PyTorch推理
    torch_input = torch.tensor(test_input)
    with torch.no_grad():
        torch_output = model(torch_input).numpy()
    
    # ONNX推理
    ort_inputs = {'input': test_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # 比较输出
    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-03, atol=1e-05)
    print("PyTorch和ONNX模型输出一致，验证成功!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='导出PyTorch模型到ONNX格式')
    parser.add_argument('--model_path', type=str, default='best_arc_quality_model.pth',
                        help='PyTorch模型路径')
    parser.add_argument('--onnx_path', type=str, default='arc_quality_model.onnx',
                        help='ONNX模型输出路径')
    parser.add_argument('--seq_length', type=int, default=500,
                        help='序列长度（默认500点）')
    
    args = parser.parse_args()
    
    export_model_to_onnx(args.model_path, args.onnx_path, args.seq_length)
