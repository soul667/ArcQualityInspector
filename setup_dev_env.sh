#!/bin/bash

# 安装额外的开发工具
pip install --no-cache-dir pytest pytest-cov ipython jupyter

# 确保脚本可执行
chmod +x start_app.sh

# 显示环境信息
echo "开发环境设置完成！"
echo "===================================="
echo "Python版本: $(python --version)"
echo "onnxruntime版本: $(python -c 'import onnxruntime; print(onnxruntime.__version__)')"
echo "===================================="
echo "如需启动应用，请运行: ./start_app.sh"
echo "或者直接运行: python src/arc_quality_inspector.py"