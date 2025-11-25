# Drone_Detection_Project/src/evaluation/evaluate_audio_finetuned_on_ood.py
import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix
from tqdm import tqdm
import pandas as pd

# Import từ các module trong src
try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
except ImportError:
    import sys
    current_dir_eval_audio_ood = os.path.dirname(os.path.abspath(__file__))
    src_dir_eval_audio_ood = os.path.dirname(current_dir_eval_audio_ood)
    project_root_eval_audio_ood = os.path.dirname(src_dir_eval_audio_ood)
    if project_root_eval_audio_ood not in sys.path: sys.path.insert(0, project_root_eval_audio_ood)
    if src_dir_eval_audio_ood not in sys.path: sys.path.insert(0, src_dir_eval_audio_ood)
    from config_loader.loader import get_config
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from evaluation.plotting_utils import plot_custom_confusion_matrix

cfg_eval_audio = get_config()

def evaluate_audio_model_on_ood(
    ood_metadata_csv_path,        # File CSV metadata của OOD
    ood_audio_segments_base_dir,  # Thư mục gốc chứa các file .wav segment OOD
    audio_model_path,             # Đường dẫn đến model audio fine-tuned .keras
    audio_scaler_path,            # Đường dẫn đến scaler .pkl đã fit cho model fine-tuned
    audio_max_len_path,           # Đường dẫn đến max_len .pkl cho model fine-tuned
    audio_label_encoder_path,     # Đường dẫn đến label_encoder .pkl cho các lớp OOD/Fusion
    report_dir_metrics,           # Thư mục lưu file report .txt
    report_dir_figures,           # Thư mục lưu ảnh CM .png
    evaluation_name_prefix="audio_ft_ood" # Tiền tố cho tên file output
):
    print(f"===== ĐÁNH GIÁ MÔ HÌNH AUDIO FINE-TUNED TRÊN TẬP OOD =====")
    print(f"Metadata OOD: {ood_metadata_csv_path}")
    print(f"Model Audio: {audio_model_path}")

    os.makedirs(report_dir_metrics, exist_ok=True)
    os.makedirs(report_dir_figures, exist_ok=True)

    # 1. Kiểm tra các file đầu vào
    required_files = [ood_metadata_csv_path, audio_model_path, audio_scaler_path, audio_max_len_path, audio_label_encoder_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"LỖI: Không tìm thấy file cần thiết: {f_path}"); return
    if not os.path.isdir(ood_audio_segments_base_dir):
        print(f"LỖI: Không tìm thấy thư mục chứa segment audio OOD: {ood_audio_segments_base_dir}"); return

    # 2. Tải các thành phần cần thiết
    try:
        test_df = pd.read_csv(ood_metadata_csv_path)
        print(f"Đã đọc {len(test_df)} mẫu từ metadata OOD.")
        
        audio_model = load_model(audio_model_path)
        print(f"  Đã tải mô hình audio từ: {audio_model_path}")
        with open(audio_scaler_path, 'rb') as f: scaler = pickle.load(f)
        print(f"  Đã tải scaler từ: {audio_scaler_path}")
        with open(audio_max_len_path, 'rb') as f: max_len = pickle.load(f)
        print(f"  Đã tải max_len: {max_len} từ: {audio_max_len_path}")
        with open(audio_label_encoder_path, 'rb') as f: label_encoder = pickle.load(f)
        print(f"  Đã tải LabelEncoder từ: {audio_label_encoder_path}")
        print(f"  Các lớp trong LabelEncoder: {list(label_encoder.classes_)}")

    except Exception as e:
        print(f"Lỗi khi tải file hoặc mô hình: {e}"); return

    all_true_labels_encoded = []
    all_pred_classes_encoded = []
    processed_count = 0

    # 3. Xử lý từng segment và dự đoán
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Evaluating Audio Fine-tuned on OOD"):
        # Tên cột trong file evaluation_test_ground_truth.csv của bạn
        audio_rel_path = row.get('path_to_audio_segment')
        true_label_text = row.get('ground_truth_label')

        if pd.isna(audio_rel_path) or pd.isna(true_label_text):
            print(f"Cảnh báo: Dòng {index} trong OOD metadata thiếu đường dẫn audio hoặc nhãn. Bỏ qua.")
            continue

        audio_abs_path = os.path.join(ood_audio_segments_base_dir, audio_rel_path)
        if not os.path.exists(audio_abs_path):
            print(f"Cảnh báo: File audio OOD không tồn tại: {audio_abs_path}. Bỏ qua.")
            continue

        try:
            # Trích xuất MFCC và xử lý
            audio_data, sr = librosa.load(audio_abs_path, sr=cfg_eval_audio.AUDIO_SAMPLE_RATE)
            if len(audio_data) < int(cfg_eval_audio.AUDIO_SEGMENT_DURATION * cfg_eval_audio.AUDIO_SAMPLE_RATE * 0.1):
                # print(f"  Segment audio {audio_rel_path} quá ngắn. Bỏ qua.")
                continue

            mfcc_feat = extract_mfcc(audio_data, cfg_eval_audio.AUDIO_SAMPLE_RATE, cfg_eval_audio.AUDIO_N_MFCC,
                                     cfg_eval_audio.AUDIO_N_FFT, cfg_eval_audio.AUDIO_HOP_LENGTH)
            
            mfcc_padded, _ = pad_features([mfcc_feat], max_len=max_len) # pad_features nhận list
            if mfcc_padded.size == 0:
                # print(f"  MFCC rỗng sau padding cho {audio_rel_path}. Bỏ qua.")
                continue

            mfcc_scaled, _ = scale_features(mfcc_padded, scaler=scaler)
            if mfcc_scaled.size == 0:
                # print(f"  MFCC rỗng sau scaling cho {audio_rel_path}. Bỏ qua.")
                continue
            
            # Dự đoán
            # model.predict nhận batch, mfcc_scaled có shape (1, max_len, n_mfcc)
            pred_probs = audio_model.predict(mfcc_scaled, verbose=0)[0]
            pred_class_encoded = np.argmax(pred_probs)
            
            # Mã hóa nhãn ground truth
            try:
                true_label_encoded = label_encoder.transform([str(true_label_text).strip()])[0]
            except ValueError:
                print(f"Cảnh báo: Nhãn ground truth '{true_label_text}' không có trong LabelEncoder. Bỏ qua mẫu này.")
                continue # Bỏ qua nếu nhãn GT không hợp lệ

            all_true_labels_encoded.append(true_label_encoded)
            all_pred_classes_encoded.append(pred_class_encoded)
            processed_count += 1

        except Exception as e_process:
            print(f"Lỗi khi xử lý hoặc dự đoán file audio OOD {audio_rel_path}: {e_process}")
            continue
            
    if processed_count == 0 or not all_true_labels_encoded:
        print("Không có mẫu OOD nào được xử lý thành công. Không thể tạo báo cáo.")
        return

    print(f"\nĐã xử lý và dự đoán thành công cho {processed_count} segment audio OOD.")

    # 4. Tạo và Lưu Báo cáo
    report_filename_txt = os.path.join(report_dir_metrics, f"{evaluation_name_prefix}_classification_report.txt")
    cm_filename_png = os.path.join(report_dir_figures, f"{evaluation_name_prefix}_confusion_matrix.png")
    
    # Danh sách tên lớp từ LabelEncoder đã tải (phải khớp với các lớp trong OOD metadata)
    class_names_for_report = list(label_encoder.classes_)
    # Tạo danh sách các ID lớp từ 0 đến N-1 cho tham số 'labels'
    labels_idx_for_report = list(range(len(class_names_for_report)))


    print(f"\n--- Classification Report (Audio Fine-tuned trên OOD - {processed_count} mẫu) ---")
    report_str = classification_report(
        all_true_labels_encoded,
        all_pred_classes_encoded,
        labels=labels_idx_for_report, # Sử dụng list các ID số nguyên
        target_names=class_names_for_report,
        zero_division=0
    )
    print(report_str)
    with open(report_filename_txt, 'w') as f:
        f.write(f"Classification Report for Fine-tuned Audio Model on OOD Test Set ({evaluation_name_prefix})\n")
        f.write(f"Total segments evaluated: {processed_count}\n")
        f.write("="*80 + "\n")
        f.write(report_str)
    print(f"Đã lưu Classification Report vào: {report_filename_txt}")

    plot_custom_confusion_matrix(
        all_true_labels_encoded,
        all_pred_classes_encoded,
        classes=class_names_for_report,
        filename=cm_filename_png,
        title=f"CM - Audio Fine-tuned on OOD ({evaluation_name_prefix})"
    )
    print(f"Đã lưu Confusion Matrix vào: {cm_filename_png}")

    # (Tùy chọn) Lưu lại các mảng dự đoán và nhãn thực tế nếu cần phân tích sâu hơn
    pred_pkl_path = os.path.join(report_dir_metrics, f"{evaluation_name_prefix}_predictions_encoded.pkl")
    true_pkl_path = os.path.join(report_dir_metrics, f"{evaluation_name_prefix}_true_labels_encoded.pkl")
    with open(pred_pkl_path, 'wb') as f: pickle.dump(all_pred_classes_encoded, f)
    with open(true_pkl_path, 'wb') as f: pickle.dump(all_true_labels_encoded, f)
    print(f"Đã lưu mảng dự đoán vào: {pred_pkl_path}")
    print(f"Đã lưu mảng nhãn thực tế vào: {true_pkl_path}")


    print("===== ĐÁNH GIÁ MÔ HÌNH AUDIO FINE-TUNED TRÊN OOD HOÀN TẤT =====")


if __name__ == '__main__':
    # Đảm bảo các thư mục output được tạo
    if hasattr(cfg_eval_audio, 'ensure_output_directories') and callable(cfg_eval_audio.ensure_output_directories):
        cfg_eval_audio.ensure_output_directories()
    else:
        os.makedirs(cfg_eval_audio.EVALUATION_OOD_REPORTS_METRICS_DIR, exist_ok=True)
        os.makedirs(cfg_eval_audio.EVALUATION_REPORTS_METRICS_DIR, exist_ok=True)

    evaluate_audio_model_on_ood(
        ood_metadata_csv_path=cfg_eval_audio.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE,
        ood_audio_segments_base_dir=cfg_eval_audio.EVALUATION_TEST_SEGMENTS_BASE_DIR, # Thư mục chứa các thư mục con theo lớp, rồi mới đến audio_segments
        audio_model_path=cfg_eval_audio.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH,
        audio_scaler_path=cfg_eval_audio.AUDIO_YOUTUBE_SCALER_PATH,
        audio_max_len_path=cfg_eval_audio.AUDIO_YOUTUBE_MAX_LEN_PATH,
        audio_label_encoder_path=cfg_eval_audio.AUDIO_YOUTUBE_LABEL_ENCODER_PATH, # Encoder đã fit trên các lớp của YT fine-tune (nên là 5 lớp)
        report_dir_metrics=cfg_eval_audio.EVALUATION_OOD_REPORTS_METRICS_DIR,
        report_dir_figures=cfg_eval_audio.EVALUATION_REPORTS_FIGURES_DIR,
        evaluation_name_prefix="audio_youtube_ft_on_ood" # Tiền tố để phân biệt file output
    )