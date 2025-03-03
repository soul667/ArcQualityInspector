# ArcQualityInspector

这是一个基于RNN的圆弧质量检测系统，具备多样化数据生成和交互式UI界面，可用于检查和标注圆弧质量。

## 功能特点

1. **多样化的缺陷模拟**：
   - 噪声增强
   - 局部变形
   - 半径变化
   - 高频波纹
   - 不连续性（断点）
   - 锯齿状缺陷
   - 过度平滑

2. **交互式UI界面**：
   - 图形化显示圆弧
   - 人工判断与模型预测对比
   - 实时统计分析
   - 数据导出功能

3. **ONNX部署**：
   - 无需PyTorch环境
   - 轻量级部署
   - 快速推理

## 安装说明

### 基本环境需求

```
numpy>=1.19.0
matplotlib>=3.3.0
onnxruntime>=1.8.0
flask>=2.0.0
scipy>=1.6.0
```

### 安装步骤

1. 克隆代码库

```bash
git clone https://github.com/username/arc-quality-inspector.git
cd arc-quality-inspector
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 准备ONNX模型
   - 如果有PyTorch模型，可使用以下命令转换:
   ```bash
   python export_to_onnx.py --model_path models/best_model.pth --onnx_path models/arc_model.onnx
   ```

## 使用说明

### 启动系统

```bash
python arc_quality_inspector.py
```

然后在浏览器中访问 `http://localhost:5000`

### 系统使用流程

1. **生成圆弧**:
   - 随机生成：默认
   - 有缺陷圆弧：点击"生成有缺陷圆弧"
   - 无缺陷圆弧：点击"生成无缺陷圆弧"

2. **检查圆弧**:
   - 查看圆弧图像
   - 参考模型预测结果
   - 做出您的判断：接受或拒绝

3. **统计与分析**:
   - 实时更新模型和人工判断的准确率
   - 查看混淆矩阵详细分析
   - 导出会话数据进行进一步分析

### 提示

- 缺陷区域通常显示为红色高亮区域
- 您可以根据专业知识覆盖模型的判断
- 定期导出数据以免丢失

## 文件说明

- `enhanced_arc_generator.py`: 增强的圆弧数据生成器
- `arc_quality_inspector.py`: Flask Web应用和UI界面
- `export_to_onnx.py`: PyTorch模型转ONNX工具
- `arc_quality_model.onnx`: 预训练的ONNX模型文件
- `templates/index.html`: UI界面模板

## 自定义配置

您可以在`enhanced_arc_generator.py`中修改以下参数来自定义数据生成:

- 圆弧长度
- 缺陷类型及其参数
- 噪声水平

## 许可证

MIT