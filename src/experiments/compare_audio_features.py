# Drone_Detection_Project/src/experiments/compare_audio_features.py
"""
Script thực nghiệm so sánh các đặc trưng audio: MFCC vs ZCR vs RMSE
Chứng minh MFCC vượt trội hơn các đặc trưng cơ bản
"""

import os
import sys
import pickle
import numpy as np
import librosa
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Thêm đường dẫn project vào sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import project_config as cfg
    from src.data_processing.audio_utils import pad_features, scale_features
except ImportError:
    print("Lỗi import config. Vui lòng kiểm tra đường dẫn.")
    sys.exit(1)


# ==================== HÀM TRÍCH XUẤT ĐẶC TRƯNG ====================

def extract_mfcc(audio_segment, sr, n_mfcc, n_fft, hop_length):
    """Trích xuất MFCC features"""
    mfccs = librosa.feature.mfcc(
        y=audio_segment, 
        sr=sr, 
        n_mfcc=n_mfcc,
        n_fft=n_fft, 
        hop_length=hop_length,
        window='hamming', 
        center=False
    )
    return mfccs.T  # Shape: (timesteps, n_mfcc)


def extract_zcr(audio_segment, sr, frame_length=2048, hop_length=512):
    """
    Trích xuất Zero Crossing Rate (ZCR)
    Returns: array 1D với giá trị ZCR cho mỗi frame
    """
    zcr = librosa.feature.zero_crossing_rate(
        y=audio_segment,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return zcr.T  # Shape: (timesteps, 1)


def extract_rmse(audio_segment, sr, frame_length=2048, hop_length=512):
    """
    Trích xuất Root Mean Square Energy (RMSE)
    Returns: array 1D với giá trị RMSE cho mỗi frame
    """
    rms = librosa.feature.rms(
        y=audio_segment,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return rms.T  # Shape: (timesteps, 1)


# ==================== HÀM XỬ LÝ DỮ LIỆU ====================

def load_and_extract_features(base_path, classes, feature_type='mfcc', 
                             segment_duration=1.0, sr=44100, 
                             n_mfcc=13, n_fft=1102, hop_length=441):
    """
    Load audio files và trích xuất đặc trưng theo loại
    
    Args:
        feature_type: 'mfcc', 'zcr', hoặc 'rmse'
    
    Returns:
        features_list: List các feature arrays
        labels_list: List các nhãn
    """
    features_list = []
    labels_list = []
    
    print(f"\n=== Trích xuất đặc trưng: {feature_type.upper()} ===")
    
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"  Cảnh báo: Thư mục {class_path} không tồn tại. Bỏ qua.")
            continue
        
        wav_files = glob.glob(os.path.join(class_path, '*.wav'))
        print(f"  Lớp {class_name}: {len(wav_files)} files")
        
        for file_path in tqdm(wav_files, desc=f"  Processing {class_name}"):
            try:
                audio, current_sr = librosa.load(file_path, sr=sr)
                
                # Chia thành segments
                segment_samples = int(segment_duration * sr)
                for start in range(0, len(audio) - segment_samples + 1, segment_samples):
                    segment = audio[start:start + segment_samples]
                    if len(segment) == segment_samples:
                        # Trích xuất đặc trưng theo loại
                        if feature_type == 'mfcc':
                            feature = extract_mfcc(segment, sr, n_mfcc, n_fft, hop_length)
                        elif feature_type == 'zcr':
                            feature = extract_zcr(segment, sr)
                        elif feature_type == 'rmse':
                            feature = extract_rmse(segment, sr)
                        else:
                            raise ValueError(f"Loại đặc trưng không hợp lệ: {feature_type}")
                        
                        features_list.append(feature)
                        labels_list.append(class_name)
                        
            except Exception as e:
                print(f"    Lỗi xử lý file {file_path}: {e}")
                continue
    
    print(f"  Tổng số segments: {len(features_list)}")
    return features_list, labels_list


def prepare_features_for_model(features_list, feature_type='mfcc', max_len=None, scaler=None):
    """
    Chuẩn bị features cho mô hình: padding và scaling
    
    Args:
        features_list: List các feature arrays
        feature_type: 'mfcc', 'zcr', hoặc 'rmse'
        max_len: Độ dài tối đa (None nếu cần tính)
        scaler: Scaler đã fit (None nếu cần fit mới)
    
    Returns:
        features_processed: Array đã xử lý
        max_len: Độ dài tối đa
        scaler: Scaler đã fit
    """
    if not features_list:
        return np.array([]), 0, None
    
    # Tính max_len nếu chưa có
    if max_len is None:
        max_len = max(feat.shape[0] for feat in features_list if feat.ndim > 0)
        print(f"  Max length: {max_len}")
    
    # Padding
    padded_features = []
    for feat in features_list:
        if feat.shape[0] < max_len:
            # Đệm bằng 0
            if feat.ndim == 2:
                padded = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant')
            else:
                # Nếu là 1D, reshape thành 2D trước
                feat_2d = feat.reshape(-1, 1) if feat.ndim == 1 else feat
                padded = np.pad(feat_2d, ((0, max_len - feat_2d.shape[0]), (0, 0)), mode='constant')
        else:
            padded = feat[:max_len, :] if feat.ndim == 2 else feat[:max_len].reshape(-1, 1)
        padded_features.append(padded)
    
    features_array = np.array(padded_features)
    print(f"  Features shape sau padding: {features_array.shape}")
    
    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        # Reshape để fit scaler
        n_samples, n_timesteps, n_features = features_array.shape
        features_reshaped = features_array.reshape(-1, n_features)
        scaler.fit(features_reshaped)
        print(f"  Scaler đã fit trên {n_samples * n_timesteps} samples")
    
    # Transform
    n_samples, n_timesteps, n_features = features_array.shape
    features_reshaped = features_array.reshape(-1, n_features)
    features_scaled_reshaped = scaler.transform(features_reshaped)
    features_scaled = features_scaled_reshaped.reshape(n_samples, n_timesteps, n_features)
    
    print(f"  Features shape sau scaling: {features_scaled.shape}")
    return features_scaled, max_len, scaler


# ==================== HÀM XÂY DỰNG MÔ HÌNH ====================

def build_model_for_feature(input_shape, num_classes, feature_type='mfcc', learning_rate=0.001):
    """
    Xây dựng mô hình phù hợp với từng loại đặc trưng
    
    Args:
        input_shape: (timesteps, features)
        num_classes: Số lớp
        feature_type: 'mfcc', 'zcr', hoặc 'rmse'
        learning_rate: Learning rate
    """
    print(f"\n=== Xây dựng mô hình cho {feature_type.upper()} ===")
    print(f"  Input shape: {input_shape}")
    
    model = Sequential([
        Input(shape=input_shape, name=f'{feature_type}_input'),
    ])
    
    # Với MFCC: sử dụng BiLSTM (giống mô hình gốc)
    if feature_type == 'mfcc':
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True), name='bilstm_1'
        ))
        model.add(Dropout(0.3, name='dropout_1'))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False), name='bilstm_2'
        ))
        model.add(Dropout(0.3, name='dropout_2'))
    
    # Với ZCR và RMSE: sử dụng LSTM đơn giản hơn (vì chỉ có 1 feature)
    else:
        model.add(LSTM(32, return_sequences=True, name='lstm_1'))
        model.add(Dropout(0.3, name='dropout_1'))
        model.add(LSTM(32, return_sequences=False, name='lstm_2'))
        model.add(Dropout(0.3, name='dropout_2'))
    
    # Lớp Dense chung
    model.add(Dense(64, activation='relu', name='dense_1'))
    model.add(Dropout(0.3, name='dropout_3'))
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ ====================

def train_and_evaluate(features, labels, feature_type, 
                      epochs=30, batch_size=32, validation_split=0.2,
                      output_dir='experiments/results'):
    """
    Huấn luyện và đánh giá mô hình với một loại đặc trưng
    """
    print(f"\n{'='*60}")
    print(f"HUẤN LUYỆN VÀ ĐÁNH GIÁ: {feature_type.upper()}")
    print(f"{'='*60}")
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Chia train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_encoded,
        test_size=validation_split,
        random_state=42,
        stratify=labels_encoded
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Xây dựng mô hình
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model_for_feature(input_shape, num_classes, feature_type)
    
    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'best_model_{feature_type}.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, 
                      mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, 
                         min_lr=1e-6, verbose=1)
    ]
    
    # Huấn luyện
    print(f"\nBắt đầu huấn luyện...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Đánh giá trên validation set
    print(f"\nĐánh giá trên validation set...")
    val_pred_probs = model.predict(X_val, verbose=0)
    val_pred_classes = np.argmax(val_pred_probs, axis=1)
    
    val_accuracy = accuracy_score(y_val, val_pred_classes)
    val_f1 = f1_score(y_val, val_pred_classes, average='weighted', zero_division=0)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  F1-Score (weighted): {val_f1:.4f}")
    
    # Classification report
    report = classification_report(
        y_val, val_pred_classes,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print(f"\nClassification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, val_pred_classes)
    
    # Lưu kết quả
    results = {
        'feature_type': feature_type,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'label_encoder': label_encoder,
        'history': history.history
    }
    
    results_path = os.path.join(output_dir, f'results_{feature_type}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nKết quả đã lưu: {results_path}")
    
    return results


# ==================== HÀM SO SÁNH KẾT QUẢ ====================

def compare_results(results_dict, output_dir='experiments/results'):
    """
    So sánh kết quả của các loại đặc trưng và tạo báo cáo
    """
    print(f"\n{'='*60}")
    print("SO SÁNH KẾT QUẢ CÁC ĐẶC TRƯNG")
    print(f"{'='*60}")
    
    # Tạo bảng so sánh
    comparison_data = []
    for feature_type, results in results_dict.items():
        comparison_data.append({
            'Feature Type': feature_type.upper(),
            'Validation Accuracy': f"{results['val_accuracy']:.4f}",
            'F1-Score (Weighted)': f"{results['val_f1']:.4f}"
        })
    
    print("\nBảng So Sánh:")
    print("-" * 60)
    for data in comparison_data:
        print(f"{data['Feature Type']:15s} | Accuracy: {data['Validation Accuracy']:8s} | F1: {data['F1-Score (Weighted)']:8s}")
    print("-" * 60)
    
    # Tìm đặc trưng tốt nhất
    best_feature = max(results_dict.items(), key=lambda x: x[1]['val_accuracy'])
    print(f"\n✓ Đặc trưng tốt nhất: {best_feature[0].upper()}")
    print(f"  Accuracy: {best_feature[1]['val_accuracy']:.4f}")
    print(f"  F1-Score: {best_feature[1]['val_f1']:.4f}")
    
    # Vẽ biểu đồ so sánh
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Biểu đồ Accuracy
    features = list(results_dict.keys())
    accuracies = [results_dict[f]['val_accuracy'] for f in features]
    f1_scores = [results_dict[f]['val_f1'] for f in features]
    
    axes[0].bar(features, accuracies, color=['#2ecc71', '#e74c3c', '#f39c12'])
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('So Sánh Accuracy')
    axes[0].set_ylim([0, 1])
    for i, acc in enumerate(accuracies):
        axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')
    
    axes[1].bar(features, f1_scores, color=['#2ecc71', '#e74c3c', '#f39c12'])
    axes[1].set_ylabel('F1-Score (Weighted)')
    axes[1].set_title('So Sánh F1-Score')
    axes[1].set_ylim([0, 1])
    for i, f1 in enumerate(f1_scores):
        axes[1].text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'feature_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nBiểu đồ so sánh đã lưu: {comparison_plot_path}")
    plt.close()
    
    # Vẽ confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (feature_type, results) in enumerate(results_dict.items()):
        cm = results['confusion_matrix']
        label_encoder = results['label_encoder']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix: {feature_type.upper()}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, 'confusion_matrices_comparison.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices đã lưu: {cm_plot_path}")
    plt.close()
    
    # Lưu báo cáo text
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BÁO CÁO SO SÁNH CÁC ĐẶC TRƯNG AUDIO\n")
        f.write("="*60 + "\n\n")
        
        f.write("BẢNG SO SÁNH:\n")
        f.write("-"*60 + "\n")
        for data in comparison_data:
            f.write(f"{data['Feature Type']:15s} | Accuracy: {data['Validation Accuracy']:8s} | F1: {data['F1-Score (Weighted)']:8s}\n")
        f.write("-"*60 + "\n\n")
        
        f.write(f"ĐẶC TRƯNG TỐT NHẤT: {best_feature[0].upper()}\n")
        f.write(f"  Accuracy: {best_feature[1]['val_accuracy']:.4f}\n")
        f.write(f"  F1-Score: {best_feature[1]['val_f1']:.4f}\n\n")
        
        f.write("\nCHI TIẾT TỪNG ĐẶC TRƯNG:\n")
        f.write("="*60 + "\n")
        for feature_type, results in results_dict.items():
            f.write(f"\n{feature_type.upper()}:\n")
            f.write(f"  Accuracy: {results['val_accuracy']:.4f}\n")
            f.write(f"  F1-Score: {results['val_f1']:.4f}\n")
            f.write(f"\n  Classification Report:\n{results['classification_report']}\n")
            f.write("-"*60 + "\n")
    
    print(f"\nBáo cáo chi tiết đã lưu: {report_path}")


# ==================== HÀM CHÍNH ====================

def main():
    """Hàm chính để chạy thực nghiệm so sánh"""
    print("="*60)
    print("THỰC NGHIỆM SO SÁNH CÁC ĐẶC TRƯNG AUDIO")
    print("MFCC vs ZCR vs RMSE")
    print("="*60)
    
    # Cấu hình
    base_path = cfg.AUDIO_RAW_DATA_PATH
    classes = cfg.AUDIO_CLASSES_ORIGINAL
    segment_duration = cfg.AUDIO_SEGMENT_DURATION
    sr = cfg.AUDIO_SAMPLE_RATE
    n_mfcc = cfg.AUDIO_N_MFCC
    n_fft = cfg.AUDIO_N_FFT
    hop_length = cfg.AUDIO_HOP_LENGTH
    
    output_dir = os.path.join(PROJECT_ROOT, 'experiments', 'feature_comparison_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Các loại đặc trưng cần so sánh
    feature_types = ['mfcc', 'zcr', 'rmse']
    results_dict = {}
    
    # Thực nghiệm với từng loại đặc trưng
    for feature_type in feature_types:
        print(f"\n\n{'#'*60}")
        print(f"XỬ LÝ ĐẶC TRƯNG: {feature_type.upper()}")
        print(f"{'#'*60}")
        
        # 1. Load và trích xuất đặc trưng
        features_list, labels_list = load_and_extract_features(
            base_path, classes, feature_type,
            segment_duration, sr, n_mfcc, n_fft, hop_length
        )
        
        if not features_list:
            print(f"  Cảnh báo: Không có features nào được trích xuất cho {feature_type}")
            continue
        
        # 2. Chuẩn bị features
        features_processed, max_len, scaler = prepare_features_for_model(
            features_list, feature_type
        )
        
        # 3. Huấn luyện và đánh giá
        results = train_and_evaluate(
            features_processed, labels_list, feature_type,
            epochs=30,  # Có thể giảm xuống để chạy nhanh hơn
            batch_size=32,
            validation_split=0.2,
            output_dir=output_dir
        )
        
        results_dict[feature_type] = results
    
    # 4. So sánh kết quả
    if len(results_dict) > 0:
        compare_results(results_dict, output_dir)
        print(f"\n{'='*60}")
        print("THỰC NGHIỆM HOÀN TẤT!")
        print(f"Kết quả đã lưu tại: {output_dir}")
        print(f"{'='*60}")
    else:
        print("\nLỗi: Không có kết quả nào để so sánh.")


if __name__ == "__main__":
    main()

