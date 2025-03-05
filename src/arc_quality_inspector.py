from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json
import uuid
import glob
import base64
import shutil
from io import BytesIO

app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

# Ensure directories exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('data_cleaned', exist_ok=True)  # New directory for cleaned data

# Store session data
session_data = {}

def load_data_samples():
    """Load samples from data directory"""
    samples = []
    for file in glob.glob('data/sample_*.json'):
        with open(file, 'r') as f:
            sample = json.load(f)
            samples.append(sample)
    return samples

def save_data_sample(file_path, sample_data):
    """Save modified sample data back to file"""
    with open(file_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

def plot_to_base64(arc_points, defect_mask=None, human_judgment=None):
    """Convert plot to base64 string"""
    plt.figure(figsize=(10, 6))
    # Plot scatter points
    x = np.arange(len(arc_points))
    plt.scatter(x, arc_points, color='blue', s=1, alpha=0.4, label='Arc Points')
    # Add light connecting line to show trend
    # plt.plot(x, arc_points, color='blue', alpha=0.2, linewidth=1)
    
    # Mark defect area if mask exists
    if defect_mask is not None:
        defect_indices = np.where(defect_mask == 1)[0]
        if len(defect_indices) > 0:
            plt.axvspan(defect_indices[0], defect_indices[-1], alpha=0.3, color='red', label='Defect Area')
    
    # Title information
    title = "Arc Data Labeling"
    
    if human_judgment is not None:
        judgment_text = "Accept" if human_judgment == 0 else "Reject"
        title += f"\nHuman Judgment: {judgment_text}"
    
    plt.title(title)
    plt.xlabel("Point Index")
    plt.ylabel("Value")
    plt.grid(True)
    if defect_mask is not None and np.any(defect_mask == 1):
        plt.legend()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

@app.route('/')
def home():
    """Homepage"""
    # Create new session
    session_id = str(uuid.uuid4())
    
    # Load all samples
    samples = load_data_samples()
    
    session_data[session_id] = {
        'arcs': samples,
        'current_index': 0,
        'total_samples': len(samples),
        'judgments': []
    }
    
    return render_template(
        'index.html',
        session_id=session_id
    )

@app.route('/next_arc', methods=['POST'])
def next_arc():
    """Get next arc data"""
    data = request.json
    session_id = data.get('session_id')
    direction = data.get('direction', 'next')  # 'next' or 'prev'
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
        
    session = session_data[session_id]
    current_index = session['current_index']
    
    # Update index based on direction
    if direction == 'prev':
        current_index = max(0, current_index - 2)  # -2 because we increment later
    
    # Check if we have more data
    if current_index >= session['total_samples']:
        return jsonify({'message': 'No more samples'}), 404
        
    # Get current sample
    sample = session['arcs'][current_index]
    
    # Increment index
    session['current_index'] = current_index + 1
    
    # Generate image
    arc_points = np.array(sample['arc_points'])
    defect_mask = np.array(sample['defect_mask'])
    img_str = plot_to_base64(arc_points, defect_mask)
    
    files = sorted(glob.glob('data/sample_*.json'))
    current_file = files[current_index]
    
    return jsonify({
        'arc_id': current_index,
        'arc_image': img_str,
        'true_defect': sample['label'],
        'file_path': current_file,
        'file_name': os.path.basename(current_file)
    })

@app.route('/submit_judgment', methods=['POST'])
def submit_judgment():
    """Submit judgment"""
    data = request.json
    session_id = data.get('session_id')
    arc_id = data.get('arc_id')
    judgment = data.get('judgment')
    update_label = data.get('update_label', False)
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    session = session_data[session_id]
    
    # Check if arc_id is valid
    if arc_id >= session['total_samples']:
        return jsonify({'error': 'Invalid arc ID'}), 400
        
    # Get arc data
    arc = session['arcs'][arc_id]
    
    # Update judgment
    arc['human_judgment'] = judgment
    session_data[session_id]['judgments'].append({
        'arc_id': arc_id,
        'true_defect': arc['label'],
        'human_judgment': judgment
    })
    
    # Update label if requested
    if update_label:
        file_path = glob.glob('data/sample_*.json')[arc_id]
        with open(file_path, 'r') as f:
            sample_data = json.load(f)
        sample_data['label'] = judgment
        save_data_sample(file_path, sample_data)
    
    # Regenerate plot
    img_str = plot_to_base64(
        np.array(arc['arc_points']), 
        np.array(arc['defect_mask']),
        judgment
    )
    
    return jsonify({
        'success': True,
        'arc_image': img_str
    })

@app.route('/delete_arc', methods=['POST'])
def delete_arc():
    """Delete arc data and file"""
    data = request.json
    session_id = data.get('session_id')
    arc_id = data.get('arc_id')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
        
    # Get file path
    files = glob.glob('data/sample_*.json')
    if arc_id >= len(files):
        return jsonify({'error': 'Invalid arc ID'}), 400
        
    file_path = os.path.abspath(files[arc_id])
    
    # Delete file
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # Reload samples after deletion
            samples = load_data_samples()
            if session_id in session_data:
                session_data[session_id]['arcs'] = samples
                session_data[session_id]['total_samples'] = len(samples)
                # Adjust current_index if needed
                if session_data[session_id]['current_index'] > len(samples):
                    session_data[session_id]['current_index'] = len(samples)
            return jsonify({'success': True, 'message': f'Deleted {file_path}'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/move_cleaned_data', methods=['POST'])
def move_cleaned_data():
    """Move processed data files to cleaned directory"""
    data = request.json
    session_id = data.get('session_id')
    current_index = data.get('current_index', 0)
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
        
    # Get all files up to current index
    files = sorted(glob.glob('data/sample_*.json'))[:current_index]
    
    moved_files = 0
    for file_path in files:
        try:
            # Create destination path
            file_name = os.path.basename(file_path)
            dest_path = os.path.join('data_cleaned', file_name)
            
            # Move file
            shutil.move(file_path, dest_path)
            moved_files += 1
        except Exception as e:
            print(f"Error moving file {file_path}: {e}")
            continue
    
    return jsonify({
        'success': True,
        'message': f'Moved {moved_files} files to data_cleaned directory'
    })

@app.route('/get_summary', methods=['POST'])
def get_summary():
    """Get session summary"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    judgments = session_data[session_id]['judgments']
    
    if not judgments:
        return jsonify({
            'message': 'No judgments yet',
            'data': {
                'total': 0,
                'human_correct': 0,
                'human_incorrect': 0,
                'confusion_matrix': {
                    'true_positive': 0,
                    'false_positive': 0,
                    'true_negative': 0,
                    'false_negative': 0
                }
            }
        })
    
    # Calculate statistics
    total = len(judgments)
    human_correct = sum(1 for j in judgments if j['human_judgment'] == j['true_defect'])
    human_incorrect = total - human_correct
    
    # Calculate confusion matrix
    confusion_matrix = {
        'true_positive': sum(1 for j in judgments if j['human_judgment'] == 1 and j['true_defect'] == 1),
        'false_positive': sum(1 for j in judgments if j['human_judgment'] == 1 and j['true_defect'] == 0),
        'true_negative': sum(1 for j in judgments if j['human_judgment'] == 0 and j['true_defect'] == 0),
        'false_negative': sum(1 for j in judgments if j['human_judgment'] == 0 and j['true_defect'] == 1)
    }
    
    return jsonify({
        'message': 'Summary data',
        'data': {
            'total': total,
            'human_correct': human_correct,
            'human_accuracy': human_correct / total if total > 0 else 0,
            'human_incorrect': human_incorrect,
            'confusion_matrix': confusion_matrix
        }
    })

@app.route('/export_session', methods=['POST'])
def export_session():
    """Export session data"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in session_data:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # Prepare export data
    export_data = {
        'session_id': session_id,
        'total_arcs': session_data[session_id]['total_samples'],
        'judgments': session_data[session_id]['judgments'],
        'timestamp': np.datetime64('now').astype(str)
    }
    
    # Save to JSON file
    filename = f"data/session_{session_id[:8]}_{export_data['timestamp'].replace(':', '-').replace(' ', '_')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'message': f"Session data saved to {filename}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
