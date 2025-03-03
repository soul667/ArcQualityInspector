from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import os
import json
import uuid
from enhanced_arc_generator import ArcDataGenerator
import base64
from io import BytesIO

app = Flask(__name__)

# 确保目录存在
os.makedirs('static/images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 初始化模型会话和数据生成器
session = None
try:
    session = ort.InferenceSession("arc_quality_model.onnx")
    print("ONNX模型加载成功")
except Exception as e:
    print(f"ONNX模型加载错误: {e}")

# 初始化数据生成器
generator = ArcDataGenerator(seq_length=100)

# 存储会话数据
session_data = {}

def detect_arc_quality(arc_points):
    """使用ONNX模型检测圆弧质量"""
    if session is None:
        # 如果模型未加载，返回随机预测
        return np.random.choice([0, 1]), np.random.random()
        
    # 准备输入数据
    input_data = arc_points.reshape(1, len(arc_points), 1).astype(np.float32)
    
    # 运行推理
    outputs = session.run(None, {'input': input_data})
    
    # 获取结果
    confidence = float(outputs[0][0][0])
    has_defect = 1 if confidence >= 0.5 else 0
    
    return has_defect, confidence

def plot_to_base64(arc_points, defect_mask=None, predicted_defect=None, confidence=None, human_judgment=None):
    """将图表转换为base64字符串"""
    plt.figure(figsize=(10, 6))
    plt.plot(arc_points, color='blue', linewidth=2, label='圆弧')
    
    # 如果有缺陷掩码，标记真实缺陷区域
    if defect_mask is not None:
        defect_indices = np.where(defect_mask == 1)[0]
        if len(defect_indices) > 0:
            plt.axvspan(defect_indices[0], defect_indices[-1], alpha=0.3, color='red', label='标注的缺陷区域')
    
    # 标题信息
    title = "圆弧质量检测"
    
    if predicted_defect is not None:
        prediction_text = "有缺陷" if predicted_defect == 1 else "无缺陷"
        title += f"\n模型预测: {prediction_text}"
        if confidence is not None:
            title += f" (置信度: {confidence:.4f})"
    
    if human_judgment is not None:
        judgment_text = "接受" if human_judgment else "拒绝"
        title += f"\n人工判断: {judgment_text}"
    
    plt.title(title)
    plt.xlabel("点索引")
    plt.ylabel("值")
    plt.grid(True)
    if defect_mask is not None and np.any(defect_mask == 1):
        plt.legend()
    
    # 转换为base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

@app.route('/')
def home():
    """主页"""
    # 创建新会话
    session_id = str(uuid.uuid4())
    session_data[session_id] = {
        'arcs': [],
        'current_index': 0,
        'total_generated': 0,
        'judgments': []
    }
    
    return render_template(
        'index.html',
        session_id=session_id
    )

@app.route('/generate_arc', methods=['POST'])
def generate_arc():
    """生成新的圆弧数据"""
    data = request.json
    session_id = data.get('session_id')
    force_defect = data.get('force_defect')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # 生成新的圆弧
    if force_defect is not None:
        force_defect = force_defect == 'true'
        sample = generator.generate_arc_sample(force_defect=force_defect)
    else:
        sample = generator.generate_arc_sample()
    
    arc_points = sample['arc_points'].tolist()
    has_defect = sample['has_defect']
    defect_mask = sample['defect_mask'].tolist()
    defect_params = sample['defect_params']
    
    # 转换缺陷参数为可序列化格式
    serializable_params = {}
    for key, value in defect_params.items():
        serializable_params[str(key)] = {
            'region': value['region'],
            'type': value['type']
        }
    
    # 使用模型预测
    predicted_defect, confidence = detect_arc_quality(sample['arc_points'])
    
    # 保存到会话数据
    arc_id = session_data[session_id]['total_generated']
    session_data[session_id]['total_generated'] += 1
    session_data[session_id]['arcs'].append({
        'id': arc_id,
        'points': arc_points,
        'true_defect': has_defect,
        'defect_mask': defect_mask,
        'defect_params': serializable_params,
        'predicted_defect': predicted_defect,
        'confidence': confidence,
        'human_judgment': None
    })
    
    # 获取图表base64
    img_str = plot_to_base64(
        sample['arc_points'], 
        sample['defect_mask'], 
        predicted_defect, 
        confidence
    )
    
    return jsonify({
        'arc_id': arc_id,
        'arc_image': img_str,
        'true_defect': has_defect,
        'predicted_defect': int(predicted_defect),
        'confidence': float(confidence)
    })

@app.route('/submit_judgment', methods=['POST'])
def submit_judgment():
    """提交人工判断"""
    data = request.json
    session_id = data.get('session_id')
    arc_id = data.get('arc_id')
    judgment = data.get('judgment')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # 查找对应的圆弧
    arc = None
    for a in session_data[session_id]['arcs']:
        if a['id'] == arc_id:
            arc = a
            break
    
    if arc is None:
        return jsonify({'error': 'Arc not found'}), 404
    
    # 更新判断
    arc['human_judgment'] = judgment
    session_data[session_id]['judgments'].append({
        'arc_id': arc_id,
        'true_defect': arc['true_defect'],
        'predicted_defect': arc['predicted_defect'],
        'human_judgment': judgment
    })
    
    # 重新生成图表
    img_str = plot_to_base64(
        np.array(arc['points']), 
        np.array(arc['defect_mask']), 
        arc['predicted_defect'], 
        arc['confidence'],
        judgment
    )
    
    return jsonify({
        'success': True,
        'arc_image': img_str
    })

@app.route('/get_summary', methods=['POST'])
def get_summary():
    """获取会话摘要"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    judgments = session_data[session_id]['judgments']
    
    if not judgments:
        return jsonify({
            'message': '尚未提交任何判断',
            'data': {
                'total': 0,
                'model_correct': 0,
                'model_incorrect': 0,
                'human_agree': 0,
                'human_disagree': 0
            }
        })
    
    # 计算统计数据
    total = len(judgments)
    model_correct = sum(1 for j in judgments if j['predicted_defect'] == j['true_defect'])
    model_incorrect = total - model_correct
    human_agree = sum(1 for j in judgments if j['human_judgment'] == j['true_defect'])
    human_disagree = total - human_agree
    
    # 计算混淆矩阵
    model_cm = {
        'true_positive': sum(1 for j in judgments if j['predicted_defect'] == 1 and j['true_defect'] == 1),
        'false_positive': sum(1 for j in judgments if j['predicted_defect'] == 1 and j['true_defect'] == 0),
        'true_negative': sum(1 for j in judgments if j['predicted_defect'] == 0 and j['true_defect'] == 0),
        'false_negative': sum(1 for j in judgments if j['predicted_defect'] == 0 and j['true_defect'] == 1)
    }
    
    human_cm = {
        'true_positive': sum(1 for j in judgments if j['human_judgment'] == 1 and j['true_defect'] == 1),
        'false_positive': sum(1 for j in judgments if j['human_judgment'] == 1 and j['true_defect'] == 0),
        'true_negative': sum(1 for j in judgments if j['human_judgment'] == 0 and j['true_defect'] == 0),
        'false_negative': sum(1 for j in judgments if j['human_judgment'] == 0 and j['true_defect'] == 1)
    }
    
    return jsonify({
        'message': '摘要数据',
        'data': {
            'total': total,
            'model_correct': model_correct,
            'model_accuracy': model_correct / total if total > 0 else 0,
            'model_incorrect': model_incorrect,
            'human_agree': human_agree,
            'human_accuracy': human_agree / total if total > 0 else 0,
            'human_disagree': human_disagree,
            'model_confusion_matrix': model_cm,
            'human_confusion_matrix': human_cm
        }
    })

@app.route('/export_session', methods=['POST'])
def export_session():
    """导出会话数据"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # 准备导出数据
    export_data = {
        'session_id': session_id,
        'total_arcs': session_data[session_id]['total_generated'],
        'judgments': session_data[session_id]['judgments'],
        'timestamp': np.datetime64('now').astype(str)
    }
    
    # 保存为JSON文件
    filename = f"data/session_{session_id[:8]}_{export_data['timestamp'].replace(':', '-').replace(' ', '_')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'message': f"会话数据已保存到 {filename}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)