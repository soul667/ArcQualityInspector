#!/bin/bash

# 确保目录存在
mkdir -p data static/images

# 如果ONNX模型不存在，显示警告
if [ ! -f "models/arc_quality_model.onnx" ]; then
    echo "警告: ONNX模型文件不存在。应用将使用随机预测。"
    echo "如需使用真实模型，请先运行PyTorch训练或导入预训练模型。"
fi

# 启动Flask应用
echo "启动ArcQualityInspector系统..."
python src/arc_quality_inspector.py