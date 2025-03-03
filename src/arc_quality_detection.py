import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

def generate_arc_data(num_samples, seq_length, radius=1.0):
    """
    生成圆弧数据
    :param num_samples: 样本数量
    :param seq_length: 序列长度（每个圆弧上的点数）
    :param radius: 圆弧半径
    :return: 特征和标签
    """
    X = []
    y = []
    
    for i in range(num_samples):
        # 生成等间隔角度
        angles = np.linspace(0, np.pi/2, seq_length)  # 生成90度的圆弧
        
        # 确定这个样本是好的圆弧还是有缺陷的圆弧
        quality = np.random.randint(0, 2)  # 0=低质量，1=高质量
        
        if quality == 1:  # 高质量圆弧
            # 添加少量随机噪声
            noise_level = np.random.uniform(0.01, 0.05)
            arc_points = radius * np.sin(angles) + np.random.normal(0, noise_level, seq_length)
        else:  # 低质量圆弧 - 添加各种缺陷
            defect_type = np.random.randint(0, 4)
            
            if defect_type == 0:  # 较大噪声
                noise_level = np.random.uniform(0.1, 0.3)
                arc_points = radius * np.sin(angles) + np.random.normal(0, noise_level, seq_length)
            elif defect_type == 1:  # 局部变形
                arc_points = radius * np.sin(angles)
                # 在随机位置添加局部变形
                defect_start = np.random.randint(0, seq_length - seq_length//5)
                defect_length = seq_length // 5
                arc_points[defect_start:defect_start+defect_length] += np.random.normal(0, 0.2, defect_length)
            elif defect_type == 2:  # 半径变化
                varying_radius = radius + np.random.normal(0, 0.1, seq_length)
                arc_points = varying_radius * np.sin(angles)
            else:  # 非圆弧形状
                arc_points = radius * np.sin(angles) + 0.2 * np.sin(3 * angles)
        
        X.append(arc_points)
        y.append(quality)
    
    return np.array(X), np.array(y)

def visualize_samples(X, y, num_samples=5):
    """
    可视化一些样本数据
    """
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i+1)
        good_idx = np.where(y == 1)[0][i]
        plt.plot(X[good_idx])
        plt.title(f"高质量圆弧 {i+1}")
        
        plt.subplot(2, num_samples, i+1+num_samples)
        bad_idx = np.where(y == 0)[0][i]
        plt.plot(X[bad_idx])
        plt.title(f"低质量圆弧 {i+1}")
    
    plt.tight_layout()
    plt.savefig('arc_samples.png')
    plt.close()

def build_rnn_model(seq_length, rnn_type='lstm', units=64, dropout_rate=0.3):
    """
    构建RNN模型
    :param seq_length: 输入序列长度
    :param rnn_type: RNN类型 ('simple', 'lstm', 'gru')
    :param units: RNN单元数量
    :param dropout_rate: Dropout率
    :return: 构建好的模型
    """
    model = Sequential()
    
    # 输入层形状为 (seq_length, 1)
    if rnn_type.lower() == 'simple':
        model.add(SimpleRNN(units, input_shape=(seq_length, 1), return_sequences=True))
    elif rnn_type.lower() == 'lstm':
        model.add(LSTM(units, input_shape=(seq_length, 1), return_sequences=True))
    else:  # GRU
        model.add(tf.keras.layers.GRU(units, input_shape=(seq_length, 1), return_sequences=True))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # 第二层RNN
    if rnn_type.lower() == 'simple':
        model.add(SimpleRNN(units // 2))
    elif rnn_type.lower() == 'lstm':
        model.add(LSTM(units // 2))
    else:  # GRU
        model.add(tf.keras.layers.GRU(units // 2))
    
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # 输出层
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def main():
    # 参数设置
    NUM_SAMPLES = 1000
    SEQ_LENGTH = 100
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # 生成数据
    X, y = generate_arc_data(NUM_SAMPLES, SEQ_LENGTH)
    
    # 可视化一些样例
    visualize_samples(X, y)
    
    # 划分训练集、验证集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 将数据重新调整为RNN所需的形状: [samples, time_steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 构建模型
    model = build_rnn_model(SEQ_LENGTH, rnn_type='lstm')
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 模型概述
    model.summary()
    
    # 早停和模型检查点
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_arc_quality_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集上的损失: {loss:.4f}")
    print(f"测试集上的准确率: {accuracy:.4f}")
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred_classes)
    print("混淆矩阵:")
    print(cm)
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_classes, target_names=['低质量', '高质量']))
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()