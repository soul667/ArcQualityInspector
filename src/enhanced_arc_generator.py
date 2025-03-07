import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class ArcDataGenerator:
    """增强的圆弧数据生成器，提供多样化的圆弧和缺陷模式"""
    
    def __init__(self, seq_length=500, radius=1.0, smoothing_window=5):
        self.seq_length = seq_length
        self.base_radius = radius
        self.smoothing_window = smoothing_window
        
    def generate_perfect_arc(self, start_angle=0, end_angle=np.pi/2, phase_shift=0):
        """生成完美的圆弧"""
        angles = np.linspace(start_angle, end_angle, self.seq_length) + phase_shift
        arc_points = self.base_radius * np.sin(angles)
        return arc_points, angles
    
    def add_base_noise(self, arc_points, noise_level=0.005):
        """添加基础噪声并平滑"""
        noisy_points = arc_points + np.random.normal(0, noise_level, self.seq_length)
        # Apply light smoothing
        return savgol_filter(noisy_points, self.smoothing_window, 2)
    
    def add_noise_defect(self, arc_points, region, noise_intensity=(0.05, 0.15)):
        """添加噪声缺陷"""
        start, end = region
        length = end - start
        noise_level = np.random.uniform(noise_intensity[0], noise_intensity[1])
        arc_points[start:end] += np.random.normal(0, noise_level, length)
        return arc_points
        
    def add_deformation_defect(self, arc_points, region, intensity=(0.1, 0.3)):
        """添加局部变形缺陷"""
        start, end = region
        length = end - start
        max_deform = np.random.uniform(intensity[0], intensity[1])
        # 使用平滑的变形
        deformation = max_deform * np.sin(np.linspace(0, np.pi, length))
        # 随机选择向上或向下变形
        if np.random.choice([True, False]):
            deformation = -deformation
        arc_points[start:end] += deformation
        return arc_points
        
    def add_radius_change_defect(self, arc_points, region, perfect_arc, min_change=0.2, max_change=0.5):
        """添加半径变化缺陷"""
        start, end = region
        length = end - start
        # 确保半径变化的绝对值在指定范围内
        radius_change_abs = np.random.uniform(min_change, max_change)
        # 随机决定正负方向
        radius_change = radius_change_abs * np.random.choice([-1, 1])
        original_values = perfect_arc[start:end]
        # 平滑过渡
        factor = 1 + radius_change * np.sin(np.linspace(0, np.pi, length))
        arc_points[start:end] = original_values * factor
        return arc_points
        
    def add_discontinuity_defect(self, arc_points, region, gap_size=(0.1, 0.3), break_prob=0.5):
        """添加不连续缺陷（断点或偏移）"""
        start, end = region
        gap = np.random.uniform(gap_size[0], gap_size[1])
        
        # 有一定概率完全断开
        if np.random.random() < break_prob:
            # 在区域中间完全断开
            mid = (start + end) // 2
            mid_width = max(2, (end - start) // 5)  # 断开宽度
            break_start = mid - mid_width // 2
            break_end = mid + mid_width // 2
            arc_points[break_start:break_end] = 0  # 完全断开，使用0值代替NaN
        else:
            # 对整个区域添加偏移
            if np.random.choice([True, False]):
                arc_points[start:end] += gap
            else:
                arc_points[start:end] -= gap
        return arc_points
        
    def add_jagged_defect(self, arc_points, region, intensity=(0.05, 0.2)):
        """添加锯齿状缺陷"""
        start, end = region
        length = end - start
        max_intensity = np.random.uniform(intensity[0], intensity[1])
        
        # 创建锯齿图案
        jagged_pattern = max_intensity * (2 * (np.arange(length) % 2) - 1)
        arc_points[start:end] += jagged_pattern
        return arc_points
    
    def add_thickness_variation_defect(self, arc_points, region, intensity=(0.05, 0.2)):
        """模拟厚度变化（通过增加上下包络线）"""
        # 这不会直接修改arc_points，而是返回上下包络
        start, end = region
        thickness = np.random.uniform(intensity[0], intensity[1])
        
        upper_env = np.copy(arc_points)
        lower_env = np.copy(arc_points)
        
        # 在有缺陷区域，厚度变化更大
        upper_env[start:end] += thickness
        lower_env[start:end] -= thickness
        
        return upper_env, lower_env
    
    def generate_arc_sample(self, force_defect=None, num_defects=None):
        """生成一个圆弧样本"""
        # 生成基础圆弧
        perfect_arc, angles = self.generate_perfect_arc()
        arc_points = self.add_base_noise(perfect_arc.copy())
        
        # 决定是否有缺陷
        has_defect = True if force_defect is True else (False if force_defect is False else np.random.choice([True, False]))
        
        # 初始化缺陷掩码和参数记录
        defect_mask = np.zeros(self.seq_length)
        defect_params = {}
        
        if has_defect:
            # 决定缺陷数量
            if num_defects is None:
                num_defects = np.random.randint(1, 3)  # 1或2个缺陷
                
            for i in range(num_defects):
                # 确定缺陷区间长度
                defect_length = np.random.randint(
                    max(5, self.seq_length // 20), 
                    max(6, self.seq_length // 5)
                )
                
                # 确定缺陷起始位置
                if i == 0:
                    defect_start = np.random.randint(0, self.seq_length - defect_length)
                else:
                    # 避免与之前的缺陷区域重叠太多
                    prev_end = defect_params[i-1]['region'][1]
                    min_start = min(prev_end, self.seq_length - defect_length)
                    if min_start >= self.seq_length - defect_length:
                        break
                    defect_start = np.random.randint(min_start, self.seq_length - defect_length)
                
                defect_end = defect_start + defect_length
                defect_region = (defect_start, defect_end)
                
                # 更新缺陷掩码
                defect_mask[defect_start:defect_end] = 1
                
                # 随机选择缺陷类型
                defect_type = np.random.choice([
                    'noise', 'deformation', 'radius_change', 
                    'discontinuity', 'jagged'
                ])
                
                # 应用相应的缺陷
                if defect_type == 'noise':
                    arc_points = self.add_noise_defect(arc_points, defect_region)
                elif defect_type == 'deformation':
                    arc_points = self.add_deformation_defect(arc_points, defect_region)
                elif defect_type == 'radius_change':
                    arc_points = self.add_radius_change_defect(arc_points, defect_region, perfect_arc)
                elif defect_type == 'discontinuity':
                    arc_points = self.add_discontinuity_defect(arc_points, defect_region)
                elif defect_type == 'jagged':
                    arc_points = self.add_jagged_defect(arc_points, defect_region)
                
                # 保存缺陷参数
                defect_params[i] = {
                    'region': defect_region,
                    'type': defect_type
                }
        
        return {
            'arc_points': arc_points,
            'has_defect': int(has_defect),
            'defect_mask': defect_mask,
            'defect_params': defect_params
        }
    
    def generate_dataset(self, num_samples, defect_ratio=0.5, save_dir=None):
        """生成数据集"""
        X = []
        y = []
        defect_masks = []
        all_params = []
        
        num_defects = int(num_samples * defect_ratio)
        
        # 生成有缺陷的样本
        for _ in range(num_defects):
            sample = self.generate_arc_sample(force_defect=True)
            X.append(sample['arc_points'])
            y.append(1)
            defect_masks.append(sample['defect_mask'])
            all_params.append(sample['defect_params'])
        
        # 生成无缺陷的样本
        for _ in range(num_samples - num_defects):
            sample = self.generate_arc_sample(force_defect=False)
            X.append(sample['arc_points'])
            y.append(0)
            defect_masks.append(sample['defect_mask'])
            all_params.append(sample['defect_params'])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        defect_masks = np.array(defect_masks)
        
        # Save data if directory is provided
        if save_dir:
            import os
            import json
            from datetime import datetime
            
            # Create data directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save each sample as individual JSON file
            for i in range(num_samples):
                sample_data = {
                    'id': i,
                    'timestamp': timestamp,
                    'arc_points': X[i].tolist(),
                    'label': int(y[i]),
                    'defect_mask': defect_masks[i].tolist(),
                    'parameters': all_params[i],
                    'metadata': {
                        'seq_length': self.seq_length,
                        'base_radius': self.base_radius
                    }
                }
                
                filename = os.path.join(save_dir, f'sample_{i:04d}_{timestamp}.json')
                with open(filename, 'w') as f:
                    json.dump(sample_data, f, indent=2)
            
            print(f"\nSaved {num_samples} samples to {save_dir}")
            print("Format: sample_XXXX_TIMESTAMP.json")
        
        return X, y, defect_masks, all_params

    def visualize_samples(self, samples, labels=None, defect_masks=None, num_samples=5):
        """可视化样本"""
        if labels is None:
            # 使用单个样本进行可视化
            plt.figure(figsize=(10, 6))
            plt.plot(samples, label='Arc Points')
            plt.title("Arc Sample")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            return
            
        plt.figure(figsize=(20, 8))
        
        # 找出缺陷和无缺陷样本的索引
        defect_indices = np.where(labels == 1)[0]
        no_defect_indices = np.where(labels == 0)[0]
        
        # 随机选择样本
        if len(defect_indices) > 0:
            defect_samples = np.random.choice(defect_indices, min(num_samples, len(defect_indices)), replace=False)
        else:
            defect_samples = []
            
        if len(no_defect_indices) > 0:    
            no_defect_samples = np.random.choice(no_defect_indices, min(num_samples, len(no_defect_indices)), replace=False)
        else:
            no_defect_samples = []
        
        # 绘制缺陷样本
        for i, idx in enumerate(defect_samples):
            if i >= num_samples:
                break
                
            plt.subplot(2, num_samples, i + 1)
            # Plot points and handle NaN values
            y = samples[idx]
            x = np.arange(len(y))
            mask = np.isfinite(y)
            # Plot every other point to reduce density
            plt.scatter(x[mask][::2], y[mask][::2], s=0.5, alpha=0.8, color='blue')
            plt.grid(True, alpha=0.3)
            
            # 如果有缺陷掩码，高亮缺陷区域
            if defect_masks is not None:
                defect_indices_for_sample = np.where(defect_masks[idx] == 1)[0]
                if len(defect_indices_for_sample) > 0:
                    plt.axvspan(defect_indices_for_sample[0], defect_indices_for_sample[-1], alpha=0.3, color='red')
            
            # Get defect type for this sample
            defect_types = []
            for defect in params[idx].values():
                defect_types.append(defect['type'])
            
            # Map defect types to numbers with explanation
            type_map = {
                'noise': 1,         # 随机噪声
                'deformation': 2,   # 局部变形
                'radius_change': 3, # 半径变化
                'discontinuity': 4, # 不连续
                'jagged': 5        # 锯齿状
            }
            
            defect_nums = [type_map[t] for t in defect_types]
            plt.title(f"Types: {defect_nums}")
            plt.xticks([])

        # Add legend for defect types
        fig = plt.gcf()
        fig.text(0.02, 0.98, "Defect Types:", fontsize=10)
        for i, (name, num) in enumerate(type_map.items()):
            fig.text(0.02, 0.95-i*0.03, f"{num}: {name}", fontsize=9)
        
        # 绘制无缺陷样本
        for i, idx in enumerate(no_defect_samples):
            if i >= num_samples:
                break
                
            plt.subplot(2, num_samples, i + 1 + num_samples)
            x = np.arange(len(samples[idx]))
            plt.scatter(x[::2], samples[idx][::2], s=0.5, alpha=0.8, color='blue')
            plt.grid(True, alpha=0.2)
            plt.title(f"Without Defect {i+1}")
            plt.xticks([])
        
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.98, hspace=0.3, wspace=0.3)
        plt.show()

# Test the data generator
if __name__ == "__main__":
    import os
    
    # Initialize generator
    generator = ArcDataGenerator(seq_length=500)
    
    # Generate and save samples to data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    X, y, masks, params = generator.generate_dataset(20000, save_dir=data_dir)
    
    # Visualize samples
    generator.visualize_samples(X, y, masks, 5)
    
    # Print defect type distribution
    defect_types = []
    for p in params:
        if p:  # If has defect parameters
            for defect in p.values():
                defect_types.append(defect['type'])
    
    if defect_types:
        from collections import Counter
        print("\nDefect Type Distribution:")
        type_counts = Counter(defect_types)
        for defect_type, count in type_counts.items():
            print(f"{defect_type}: {count}")
