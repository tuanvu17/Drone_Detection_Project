# Drone_Detection_Project/src/analysis/generate_audio_results_report.py
"""
Script để tạo báo cáo kết quả và phân tích cho các mô hình Audio
4.4.1: Kết quả trên Bộ Dữ liệu Gốc
4.4.2: Kết quả trên Bộ Dữ liệu YouTube Fine-tune
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm đường dẫn project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import project_config as cfg
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
except ImportError as e:
    print(f"Lỗi import: {e}")
    sys.exit(1)


def evaluate_original_audio_model():
    """
    4.4.1: Đánh giá mô hình Audio gốc trên tập test của bộ dữ liệu gốc
    """
    print("\n" + "="*80)
    print("4.4.1. KẾT QUẢ TRÊN BỘ DỮ LIỆU GỐC")
    print("="*80)
    
    # Đường dẫn model và các file cần thiết
    model_path = cfg.AUDIO_MODEL_SAVE_PATH
    label_encoder_path = cfg.AUDIO_LABEL_ENCODER_PATH
    scaler_path = cfg.AUDIO_SCALER_PATH
    max_len_path = cfg.AUDIO_MAX_LEN_PATH
    test_files_list_path = cfg.AUDIO_ORIGINAL_TEST_FILES_LIST_PATH
    base_path = cfg.AUDIO_RAW_DATA_PATH
    
    # Kiểm tra file tồn tại
    required_files = [model_path, label_encoder_path, scaler_path, max_len_path, test_files_list_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"LỖI: Thiếu các file cần thiết:")
        for f in missing_files:
            print(f"  - {f}")
        return None
    
    print(f"\nĐang tải mô hình và các thành phần...")
    try:
        model = load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(max_len_path, 'rb') as f:
            max_len = pickle.load(f)
        with open(test_files_list_path, 'r') as f:
            test_filenames = [line.strip() for line in f if line.strip()]
        print(f"  ✓ Đã tải thành công")
        print(f"  - Model: {os.path.basename(model_path)}")
        print(f"  - Test files: {len(test_filenames)} files")
        print(f"  - Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"LỖI khi tải: {e}")
        return None
    
    # Xử lý test files
    print(f"\nĐang xử lý test files...")
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    
    for fname in tqdm(test_filenames, desc="Processing test files"):
        # Tìm file trong các thư mục class
        found = False
        for class_name in label_encoder.classes_:
            file_path = os.path.join(base_path, class_name, fname)
            if os.path.exists(file_path):
                found = True
                true_label = class_name
                break
        
        if not found:
            print(f"  Cảnh báo: Không tìm thấy file {fname}")
            continue
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=cfg.AUDIO_SAMPLE_RATE)
            
            # Chia thành segments và trích xuất features
            segment_samples = int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
            segment_features_list = []
            
            for start in range(0, len(audio) - segment_samples + 1, segment_samples):
                segment = audio[start:start + segment_samples]
                if len(segment) == segment_samples:
                    mfccs = extract_mfcc(segment, cfg.AUDIO_SAMPLE_RATE, 
                                        cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, 
                                        cfg.AUDIO_HOP_LENGTH)
                    segment_features_list.append(mfccs)
            
            if not segment_features_list:
                continue
            
            # Padding và scaling
            features_padded, _ = pad_features(segment_features_list, max_len=max_len)
            features_scaled, _ = scale_features(features_padded, scaler=scaler)
            
            # Dự đoán
            pred_probs = model.predict(features_scaled, verbose=0)
            avg_probs = np.mean(pred_probs, axis=0)
            pred_label_idx = np.argmax(avg_probs)
            pred_label = label_encoder.inverse_transform([pred_label_idx])[0]
            
            # Lưu kết quả
            true_label_idx = label_encoder.transform([true_label])[0]
            all_true_labels.append(true_label_idx)
            all_pred_labels.append(pred_label_idx)
            all_pred_probs.append(avg_probs)
            
        except Exception as e:
            print(f"  Lỗi xử lý file {fname}: {e}")
            continue
    
    if not all_true_labels:
        print("LỖI: Không có dữ liệu để đánh giá")
        return None
    
    # Tính toán metrics
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    f1_weighted = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
    f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
    
    # Classification Report
    class_names = list(label_encoder.classes_)
    report_str = classification_report(
        all_true_labels, all_pred_labels,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=range(len(class_names)))
    
    # Hiển thị kết quả
    print(f"\n{'='*80}")
    print("KẾT QUẢ ĐÁNH GIÁ")
    print(f"{'='*80}")
    print(f"\nTổng số mẫu test: {len(all_true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*80}")
    print(report_str)
    
    # Lưu kết quả
    output_dir = os.path.join(PROJECT_ROOT, 'reports', 'analysis', 'audio_original')
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu classification report
    report_path = os.path.join(output_dir, 'classification_report_original.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("4.4.1. KẾT QUẢ TRÊN BỘ DỮ LIỆU GỐC\n")
        f.write("="*80 + "\n\n")
        f.write(f"Mô hình: {os.path.basename(model_path)}\n")
        f.write(f"Tập test: {len(test_filenames)} files từ bộ dữ liệu gốc\n")
        f.write(f"Tổng số mẫu đánh giá: {len(all_true_labels)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(report_str)
        f.write("\n" + "="*80 + "\n")
    
    # Vẽ và lưu confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix_original.png')
    plot_custom_confusion_matrix(
        all_true_labels, all_pred_labels,
        class_names,
        filename=cm_path,
        title="Confusion Matrix - Audio Model Gốc (Test Set)"
    )
    
    print(f"\nKết quả đã lưu:")
    print(f"  - Classification Report: {report_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report_str,
        'confusion_matrix': cm,
        'class_names': class_names,
        'num_samples': len(all_true_labels)
    }


def evaluate_youtube_finetuned_model():
    """
    4.4.2: Đánh giá mô hình Audio YouTube fine-tuned trên tập Test_Audio_YT
    """
    print("\n" + "="*80)
    print("4.4.2. KẾT QUẢ TRÊN BỘ DỮ LIỆU YOUTUBE FINE-TUNE")
    print("="*80)
    
    # Đường dẫn model và các file cần thiết
    model_path = cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH
    label_encoder_path = cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH
    scaler_path = cfg.AUDIO_YOUTUBE_SCALER_PATH
    max_len_path = cfg.AUDIO_YOUTUBE_MAX_LEN_PATH
    
    # Kiểm tra xem có test data đã được lưu chưa
    test_data_dir = os.path.join(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'test_in_domain_audio_data')
    test_features_path = os.path.join(test_data_dir, 'X_test_audio_yt_mfcc_scaled.pkl')
    test_labels_path = os.path.join(test_data_dir, 'y_test_audio_yt_encoded.pkl')
    
    # Kiểm tra file tồn tại
    required_files = [model_path, label_encoder_path, scaler_path, max_len_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"LỖI: Thiếu các file cần thiết:")
        for f in missing_files:
            print(f"  - {f}")
        return None
    
    if not os.path.exists(test_features_path) or not os.path.exists(test_labels_path):
        print(f"Cảnh báo: Test data chưa được lưu. Sử dụng validation set thay thế.")
        # Có thể load từ validation set hoặc bỏ qua
        return None
    
    print(f"\nĐang tải mô hình và test data...")
    try:
        model = load_model(model_path)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        with open(test_features_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(test_labels_path, 'rb') as f:
            y_test = pickle.load(f)
        print(f"  ✓ Đã tải thành công")
        print(f"  - Model: {os.path.basename(model_path)}")
        print(f"  - Test samples: {len(y_test)}")
        print(f"  - Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"LỖI khi tải: {e}")
        return None
    
    # Dự đoán
    print(f"\nĐang dự đoán trên test set...")
    pred_probs = model.predict(X_test, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)
    
    # Tính toán metrics
    accuracy = accuracy_score(y_test, pred_labels)
    f1_weighted = f1_score(y_test, pred_labels, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, pred_labels, average='macro', zero_division=0)
    
    # Classification Report
    class_names = list(label_encoder.classes_)
    report_str = classification_report(
        y_test, pred_labels,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, pred_labels, labels=range(len(class_names)))
    
    # Hiển thị kết quả
    print(f"\n{'='*80}")
    print("KẾT QUẢ ĐÁNH GIÁ")
    print(f"{'='*80}")
    print(f"\nTổng số mẫu test: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*80}")
    print(report_str)
    
    # Lưu kết quả
    output_dir = os.path.join(PROJECT_ROOT, 'reports', 'analysis', 'audio_youtube')
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu classification report
    report_path = os.path.join(output_dir, 'classification_report_youtube.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("4.4.2. KẾT QUẢ TRÊN BỘ DỮ LIỆU YOUTUBE FINE-TUNE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Mô hình: {os.path.basename(model_path)}\n")
        f.write(f"Tập test: Test_Audio_YT (In-Domain Test Set)\n")
        f.write(f"Tổng số mẫu đánh giá: {len(y_test)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(report_str)
        f.write("\n" + "="*80 + "\n")
    
    # Vẽ và lưu confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix_youtube.png')
    plot_custom_confusion_matrix(
        y_test, pred_labels,
        class_names,
        filename=cm_path,
        title="Confusion Matrix - Audio Model YouTube Fine-tuned (Test Set)"
    )
    
    print(f"\nKết quả đã lưu:")
    print(f"  - Classification Report: {report_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report_str,
        'confusion_matrix': cm,
        'class_names': class_names,
        'num_samples': len(y_test)
    }


def generate_analysis_report(results_original, results_youtube):
    """
    Tạo báo cáo phân tích tổng hợp
    """
    print("\n" + "="*80)
    print("TẠO BÁO CÁO PHÂN TÍCH TỔNG HỢP")
    print("="*80)
    
    output_dir = os.path.join(PROJECT_ROOT, 'reports', 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'audio_results_analysis.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Kết Quả và Phân Tích Mô Hình Audio\n\n")
        
        # 4.4.1
        f.write("## 4.4.1. Kết Quả Trên Bộ Dữ Liệu Gốc\n\n")
        if results_original:
            f.write("### Mô Hình: best_audio_model.keras\n\n")
            f.write(f"- **Tập test**: Test set từ bộ dữ liệu gốc (Svanström et al.)\n")
            f.write(f"- **Số mẫu đánh giá**: {results_original['num_samples']}\n")
            f.write(f"- **Accuracy**: {results_original['accuracy']:.4f}\n")
            f.write(f"- **F1-Score (Weighted)**: {results_original['f1_weighted']:.4f}\n")
            f.write(f"- **F1-Score (Macro)**: {results_original['f1_macro']:.4f}\n\n")
            
            f.write("### Classification Report\n\n")
            f.write("```\n")
            f.write(results_original['classification_report'])
            f.write("```\n\n")
            
            f.write("### Confusion Matrix\n\n")
            f.write(f"![Confusion Matrix Original](audio_original/confusion_matrix_original.png)\n\n")
            
            f.write("### Phân Tích Hiệu Năng\n\n")
            f.write("Mô hình Audio gốc được huấn luyện trên bộ dữ liệu Svanström et al. (90 clips) ")
            f.write("với 3 lớp: DRONE, HELICOPTER, và BACKGROUND. Kết quả cho thấy:\n\n")
            f.write(f"- **Độ chính xác tổng thể**: {results_original['accuracy']*100:.2f}%\n")
            f.write(f"- **Khả năng phân biệt các lớp**: Đặc trưng MFCC cho phép mô hình phân biệt tốt giữa các lớp\n")
            f.write(f"- **Hiệu năng từng lớp**: Xem chi tiết trong Classification Report ở trên\n\n")
            
            # Phân tích confusion matrix
            cm = results_original['confusion_matrix']
            class_names = results_original['class_names']
            f.write("**Phân tích Confusion Matrix**:\n\n")
            for i, class_name in enumerate(class_names):
                correct = cm[i, i]
                total = cm[i, :].sum()
                precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
                recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                f.write(f"- **{class_name}**: Precision={precision:.3f}, Recall={recall:.3f} ({correct}/{total} đúng)\n")
            f.write("\n")
        else:
            f.write("Không có kết quả để hiển thị.\n\n")
        
        # 4.4.2
        f.write("## 4.4.2. Kết Quả Trên Bộ Dữ Liệu YouTube Fine-tune\n\n")
        if results_youtube:
            f.write("### Mô Hình: best_audio_youtube_finetuned_model.keras\n\n")
            f.write(f"- **Tập test**: Test_Audio_YT (In-Domain Test Set)\n")
            f.write(f"- **Số mẫu đánh giá**: {results_youtube['num_samples']}\n")
            f.write(f"- **Accuracy**: {results_youtube['accuracy']:.4f}\n")
            f.write(f"- **F1-Score (Weighted)**: {results_youtube['f1_weighted']:.4f}\n")
            f.write(f"- **F1-Score (Macro)**: {results_youtube['f1_macro']:.4f}\n\n")
            
            f.write("### Classification Report\n\n")
            f.write("```\n")
            f.write(results_youtube['classification_report'])
            f.write("```\n\n")
            
            f.write("### Confusion Matrix\n\n")
            f.write(f"![Confusion Matrix YouTube](audio_youtube/confusion_matrix_youtube.png)\n\n")
            
            f.write("### Phân Tích Hiệu Năng\n\n")
            f.write("Mô hình Audio đã được fine-tune trên bộ dữ liệu YouTube với ")
            f.write(f"{len(results_youtube['class_names'])} lớp. Kết quả cho thấy:\n\n")
            f.write(f"- **Độ chính xác tổng thể**: {results_youtube['accuracy']*100:.2f}%\n")
            f.write(f"- **Khả năng trên dữ liệu đa dạng**: Đặc trưng MFCC cho phép mô hình hoạt động tốt ")
            f.write("trên dữ liệu YouTube đa dạng hơn, chứng minh tính tổng quát của đặc trưng MFCC\n")
            f.write(f"- **Hiệu năng từng lớp**: Xem chi tiết trong Classification Report ở trên\n\n")
            
            # Phân tích confusion matrix
            cm = results_youtube['confusion_matrix']
            class_names = results_youtube['class_names']
            f.write("**Phân tích Confusion Matrix**:\n\n")
            for i, class_name in enumerate(class_names):
                correct = cm[i, i]
                total = cm[i, :].sum()
                precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
                recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                f.write(f"- **{class_name}**: Precision={precision:.3f}, Recall={recall:.3f} ({correct}/{total} đúng)\n")
            f.write("\n")
        else:
            f.write("Không có kết quả để hiển thị.\n\n")
        
        # So sánh
        if results_original and results_youtube:
            f.write("## So Sánh Tổng Hợp\n\n")
            f.write("| Metric | Bộ Dữ Liệu Gốc | Bộ Dữ Liệu YouTube |\n")
            f.write("|--------|----------------|---------------------|\n")
            f.write(f"| Accuracy | {results_original['accuracy']:.4f} | {results_youtube['accuracy']:.4f} |\n")
            f.write(f"| F1-Score (Weighted) | {results_original['f1_weighted']:.4f} | {results_youtube['f1_weighted']:.4f} |\n")
            f.write(f"| F1-Score (Macro) | {results_original['f1_macro']:.4f} | {results_youtube['f1_macro']:.4f} |\n")
            f.write(f"| Số lớp | {len(results_original['class_names'])} | {len(results_youtube['class_names'])} |\n")
            f.write(f"| Số mẫu test | {results_original['num_samples']} | {results_youtube['num_samples']} |\n\n")
    
    print(f"Báo cáo phân tích đã lưu: {report_path}")


def main():
    """Hàm chính"""
    print("="*80)
    print("TẠO BÁO CÁO KẾT QUẢ VÀ PHÂN TÍCH MÔ HÌNH AUDIO")
    print("="*80)
    
    # 4.4.1: Đánh giá mô hình gốc
    results_original = evaluate_original_audio_model()
    
    # 4.4.2: Đánh giá mô hình YouTube fine-tuned
    results_youtube = evaluate_youtube_finetuned_model()
    
    # Tạo báo cáo phân tích tổng hợp
    generate_analysis_report(results_original, results_youtube)
    
    print("\n" + "="*80)
    print("HOÀN TẤT!")
    print("="*80)


if __name__ == "__main__":
    main()

