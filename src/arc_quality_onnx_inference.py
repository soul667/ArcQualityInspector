import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import time

def generate_test_arc(seq_length=500, has_defect=None):
    """Generate test arc data"""
    # Generate evenly spaced angles
    angles = np.linspace(0, np.pi/2, seq_length)
    radius = 1.0
    
    # Generate perfect arc
    perfect_arc = radius * np.sin(angles)
    
    # Add small base noise
    base_noise = np.random.normal(0, 0.001, seq_length)
    arc_points = perfect_arc + base_noise
    
    # Determine if defective
    if has_defect is None:
        has_defect = np.random.choice([0, 1])
    
    # Initialize defect mask
    defect_mask = np.zeros(seq_length)
    
    if has_defect:
        # Determine defect interval length (10% to 30% of sequence length)
        defect_length = np.random.randint(seq_length // 10, seq_length // 3)
        
        # Determine defect start position
        defect_start = np.random.randint(0, seq_length - defect_length)
        defect_end = defect_start + defect_length
        
        # Update defect mask
        defect_mask[defect_start:defect_end] = 1
        
        # Randomly select defect type
        defect_type = np.random.randint(0, 4)
        
        if defect_type == 0:  # Increased noise
            noise_level = np.random.uniform(0.1, 0.3)
            arc_points[defect_start:defect_end] += np.random.normal(0, noise_level, defect_length)
        
        elif defect_type == 1:  # Local deformation
            deformation = np.random.normal(0, 0.2, defect_length)
            arc_points[defect_start:defect_end] += deformation
        
        elif defect_type == 2:  # Local radius change
            radius_change = np.random.uniform(-0.3, 0.3)
            original_values = perfect_arc[defect_start:defect_end]
            arc_points[defect_start:defect_end] = original_values * (1 + radius_change)
        
        else:  # High frequency ripple
            freq_factor = np.random.randint(3, 8)
            local_angles = np.linspace(0, np.pi * freq_factor, defect_length)
            wave_amplitude = np.random.uniform(0.05, 0.15)
            arc_points[defect_start:defect_end] += wave_amplitude * np.sin(local_angles)
    
    return arc_points, has_defect, defect_mask

def detect_arc_quality_onnx(session, arc_points):
    """Detect arc quality using ONNX model"""
    # Prepare input data - note data type must be float32
    input_data = arc_points.reshape(1, len(arc_points), 1).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {'input': input_data})
    
    # Get results
    confidence = outputs[0][0][0]
    has_defect = 1 if confidence >= 0.5 else 0
    
    return has_defect, confidence

def visualize_detection(arc_points, true_defect, predicted_defect, confidence, defect_mask=None):
    """Visualize detection results"""
    plt.figure(figsize=(10, 6))
    plt.plot(arc_points, color='blue', label='Arc')
    
    # If defect mask exists, mark actual defect area
    if defect_mask is not None and true_defect == 1:
        defect_indices = np.where(defect_mask == 1)[0]
        if len(defect_indices) > 0:
            plt.axvspan(defect_indices[0], defect_indices[-1], alpha=0.3, color='red', label='Actual Defect Area')
    
    # Get prediction result text
    prediction_text = "Defective" if predicted_defect == 1 else "Non-defective"
    truth_text = "Defective" if true_defect == 1 else "Non-defective"
    
    # Set title and labels
    plt.title(f"Arc Quality Detection\nPrediction: {prediction_text} (Confidence: {confidence:.4f}), Actual: {truth_text}")
    plt.xlabel("Point Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Start timing
    start_time = time.time()
    
    # Load ONNX model
    try:
        # Create ONNX Runtime inference session
        session = ort.InferenceSession("arc_quality_model.onnx")
        print("ONNX model loaded successfully!")
    except Exception as e:
        print(f"ONNX model loading failed: {e}")
        return
    
    # Print model loading time
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.2f} seconds")
    
    total_inference_time = 0
    correct_predictions = 0
    
    # Generate and detect multiple test samples
    for i in range(5):
        # Generate test sample
        arc_points, true_defect, defect_mask = generate_test_arc()
        
        # Detect quality (timing)
        inference_start = time.time()
        predicted_defect, confidence = detect_arc_quality_onnx(session, arc_points)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        
        # Check if prediction is correct
        if predicted_defect == true_defect:
            correct_predictions += 1
        
        # Print results
        print(f"\nTest sample {i+1}:")
        print(f"Actual state: {'Defective' if true_defect == 1 else 'Non-defective'}")
        print(f"Prediction: {'Defective' if predicted_defect == 1 else 'Non-defective'} (Confidence: {confidence:.4f})")
        print(f"Inference time: {inference_time:.4f} seconds")
        
        # Visualize results
        visualize_detection(arc_points, true_defect, predicted_defect, confidence, defect_mask)
    
    # Print statistics
    print("\nTest Statistics:")
    print(f"Accuracy: {correct_predictions/5:.2%}")
    print(f"Average inference time: {total_inference_time/5:.4f} seconds")
    print("All tests completed!")

if __name__ == "__main__":
    main()
