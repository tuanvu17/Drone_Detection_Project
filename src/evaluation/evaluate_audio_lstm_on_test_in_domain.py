# Drone_Detection_Project/src/evaluation/evaluate_audio_lstm_on_test_in_domain.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Buộc TensorFlow chạy trên CPU

import glob
import librosa
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.video.io.VideoFileClip import VideoFileClip # Để trích xuất audio từ video

# --- Thêm đường dẫn gốc của dự án vào sys.path ---
try:
    from src.config_loader.loader import get_config
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
except ImportError:
    import sys
    current_dir_eval_ds = os.path.dirname(os.path.abspath(__file__))
    src_dir_eval_ds = os.path.dirname(current_dir_eval_ds)
    project_root_eval_ds = os.path.dirname(src_dir_eval_ds)
    if project_root_eval_ds not in sys.path: sys.path.insert(0, project_root_eval_ds)
    if src_dir_eval_ds not in sys.path: sys.path.insert(0, src_dir_eval_ds)

    from config_loader.loader import get_config
    from evaluation.plotting_utils import plot_custom_confusion_matrix
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features

cfg = get_config()

# --- Helper Functions ---
def get_audio_segment_from_video_for_evaluation(original_video_path, segment_id_str, segment_duration_seconds, target_sr):
    """
    Trích xuất một đoạn audio 1 giây từ video gốc cho việc đánh giá.
    original_video_path: Đường dẫn đến file video gốc.
    segment_id_str: ID của segment, ví dụ "DRONE_video1_seg00001", dùng để lấy index.
    segment_duration_seconds: Độ dài của mỗi segment tính bằng giây.
    target_sr: Tần số lấy mẫu mục tiêu.
    """
    temp_audio_filename = f"temp_eval_audio_{os.path.basename(original_video_path)}_{segment_id_str.replace('/', '_')}.wav"
    temp_audio_path = os.path.join(cfg.INTERIM_YOUTUBE_AUDIO_DIR, temp_audio_filename) # Sử dụng thư mục tạm đã định nghĩa
    os.makedirs(cfg.INTERIM_YOUTUBE_AUDIO_DIR, exist_ok=True)

    audio_segment_data = None
    try:
        if not os.path.exists(original_video_path):
            # print(f"  CẢNH BÁO: Video gốc không tồn tại '{original_video_path}'")
            return None

        with VideoFileClip(original_video_path) as video_clip:
            if video_clip.audio is None:
                # print(f"    Video {os.path.basename(original_video_path)} không có track audio.")
                return None
            video_clip.audio.write_audiofile(temp_audio_path, fps=target_sr, logger=None, verbose=False)

        full_audio_data, sr_loaded = librosa.load(temp_audio_path, sr=target_sr)
        if sr_loaded != target_sr:
            # print(f"    CẢNH BÁO: Audio được load với SR {sr_loaded} thay vì {target_sr} từ {temp_audio_path}")
            pass

        # Trích xuất segment_index từ segment_id_str (ví dụ: 'DRONE/DRONE_video1_seg00000_audio_mfcc.npy' -> 0)
        # Hoặc nếu metadata Test In-Domain lưu segment_index trực tiếp, thì dùng nó.
        # Giả định metadata Test In-Domain sẽ có cột 'video_segment_index'
        # Nếu không, cần logic khác để lấy index từ segment_id_from_metadata
        # Hiện tại, đang dựa vào việc metadata có 'video_segment_index'
        # Nếu không có, đoạn này sẽ cần sửa để parse từ segment_id_from_metadata
        # Ví dụ, nếu segment_id_from_metadata là "SOMECLASS_video1_seg00005"
        try:
            segment_index = int(segment_id_str.split('_seg')[-1]) # Giả định cấu trúc tên này
        except ValueError:
             # Thử một cách parse khác nếu có, hoặc báo lỗi
            print(f"Lỗi parse segment index từ {segment_id_str}")
            return None


        start_sample = int(segment_index * segment_duration_seconds * target_sr)
        end_sample = int((segment_index + 1) * segment_duration_seconds * target_sr)

        if start_sample < len(full_audio_data):
            audio_segment_data = full_audio_data[start_sample:end_sample]
            # Kiểm tra độ dài tối thiểu, ví dụ 80% của segment_duration
            min_len_samples = int(segment_duration_seconds * target_sr * 0.80)
            if len(audio_segment_data) < min_len_samples:
                # print(f"    Segment audio {segment_index} từ {os.path.basename(original_video_path)} quá ngắn ({len(audio_segment_data)} samples). Min_len: {min_len_samples}")
                return None
        else:
            # print(f"    Segment audio index {segment_index} (start_sample {start_sample}) vượt quá độ dài audio ({len(full_audio_data)}) của video {os.path.basename(original_video_path)}.")
            return None

    except Exception as e:
        # print(f"    Lỗi trích xuất audio segment từ {original_video_path} cho segment {segment_id_str}: {e}")
        return None
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass # Bỏ qua nếu không xóa được file tạm
    return audio_segment_data


def get_audio_model_prediction(audio_model_keras, audio_segment_data_raw,
                               scaler_audio, max_len_audio,
                               audio_model_native_label_encoder, # Encoder mà audio_model được huấn luyện
                               target_evaluation_label_encoder, # Encoder chung để đánh giá (ví dụ: fusion_le)
                               default_class_in_eval_space="BACKGROUND"):
    """
    Tiền xử lý audio segment thô và lấy dự đoán từ mô hình audio Keras.
    Sau đó ánh xạ dự đoán về không gian nhãn của target_evaluation_label_encoder.
    """
    # Ngưỡng tối thiểu cho độ dài audio segment (ví dụ, 50% của 1 giây)
    min_required_samples = int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.5)
    if audio_segment_data_raw is None or len(audio_segment_data_raw) < min_required_samples:
        try: return target_evaluation_label_encoder.transform([default_class_in_eval_space])[0], 0.0
        except: return -1, 0.0 # Lỗi nghiêm trọng

    try:
        mfcc_s = extract_mfcc(audio_segment_data_raw, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
        mfcc_p, _ = pad_features([mfcc_s], max_len=max_len_audio) # Sử dụng max_len từ quá trình fine-tune audio
        if mfcc_p.size == 0:
            try: return target_evaluation_label_encoder.transform([default_class_in_eval_space])[0], 0.0
            except: return -1, 0.0

        mfcc_sc, _ = scale_features(mfcc_p, scaler=scaler_audio) # Sử dụng scaler từ quá trình fine-tune audio

        audio_probs = audio_model_keras.predict(mfcc_sc, verbose=0)[0]
        audio_pred_idx_native = np.argmax(audio_probs) # Index trong không gian lớp của audio_model_native_label_encoder
        confidence = float(audio_probs[audio_pred_idx_native])

        # Lấy tên lớp từ không gian lớp gốc của mô hình audio
        if audio_pred_idx_native < len(audio_model_native_label_encoder.classes_):
            predicted_class_name_native = audio_model_native_label_encoder.classes_[audio_pred_idx_native]
        else:
            # print(f"  CẢNH BÁO: Index dự đoán {audio_pred_idx_native} nằm ngoài các lớp của Audio Native LabelEncoder.")
            predicted_class_name_native = default_class_in_eval_space # Mặc định nếu có lỗi

        # Ánh xạ tên lớp đó sang không gian nhãn của target_evaluation_label_encoder
        if predicted_class_name_native in target_evaluation_label_encoder.classes_:
            return target_evaluation_label_encoder.transform([predicted_class_name_native])[0], confidence
        else:
            # print(f"  CẢNH BÁO: Lớp dự đoán '{predicted_class_name_native}' từ Audio Model không có trong Target Evaluation LabelEncoder. Mặc định BACKGROUND.")
            try: return target_evaluation_label_encoder.transform([default_class_in_eval_space])[0], 0.0
            except: return -1, 0.0

    except Exception as e_audio_pred:
        # print(f"  Lỗi trong get_audio_model_prediction: {e_audio_pred}")
        try: return target_evaluation_label_encoder.transform([default_class_in_eval_space])[0], 0.0
        except: return -1, 0.0

def evaluate_and_report_single_model(y_true_encoded, y_pred_encoded,
                                     model_full_name,
                                     target_class_names_for_report,
                                     output_cm_path,
                                     output_report_path):
    # ... (Nội dung hàm này giữ nguyên như trong file evaluate_models_on_in-domain.py) ...
    print(f"\n--- Đánh giá Mô hình: {model_full_name} ---")
    if len(y_true_encoded) == 0 or len(y_pred_encoded) == 0 :
        print(f"  CẢNH BÁO: Không có dữ liệu để đánh giá cho {model_full_name}.")
        return
    if len(y_true_encoded) != len(y_pred_encoded):
        print(f"  CẢNH BÁO: Số lượng nhãn thực tế ({len(y_true_encoded)}) và dự đoán ({len(y_pred_encoded)}) không khớp cho {model_full_name}.")
        min_len = min(len(y_true_encoded), len(y_pred_encoded))
        y_true_encoded = np.array(y_true_encoded[:min_len]) # Chuyển sang array nếu chưa
        y_pred_encoded = np.array(y_pred_encoded[:min_len])
        if min_len == 0: print("  Không còn dữ liệu sau khi điều chỉnh độ dài."); return
    
    # Lọc các giá trị -1 (lỗi) trước khi tính toán
    valid_indices = np.where((np.array(y_true_encoded) != -1) & (np.array(y_pred_encoded) != -1))[0]
    if len(valid_indices) == 0:
        print(f"  Không có cặp nhãn hợp lệ nào để đánh giá cho {model_full_name}.")
        return
    y_true_final = np.array(y_true_encoded)[valid_indices]
    y_pred_final = np.array(y_pred_encoded)[valid_indices]

    if len(y_true_final) == 0 :
        print(f"  Không có dữ liệu hợp lệ sau khi lọc lỗi cho {model_full_name}.")
        return

    accuracy = accuracy_score(y_true_final, y_pred_final)
    f1_weighted = f1_score(y_true_final, y_pred_final, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_final, y_pred_final, average='macro', zero_division=0)

    # Đảm bảo labels là list các số nguyên tương ứng với target_class_names_for_report
    unique_labels_present = sorted(list(set(y_true_final) | set(y_pred_final)))
    report_labels = [l for l in unique_labels_present if l < len(target_class_names_for_report)]
    report_target_names = [target_class_names_for_report[l] for l in report_labels]

    if not report_labels: # Nếu không có nhãn nào hợp lệ (ví dụ tất cả đều là -1)
        print(f"  Không có nhãn hợp lệ để tạo classification report cho {model_full_name}")
        report_str = "No valid labels to generate report."
    else:
        report_str = classification_report(y_true_final, y_pred_final,
                                       target_names=report_target_names, # Chỉ dùng tên lớp có trong dữ liệu
                                       labels=report_labels, # Chỉ dùng index lớp có trong dữ liệu
                                       zero_division=0)
    print(f"Accuracy: {accuracy:.4f}"); print(f"F1-Weighted: {f1_weighted:.4f}"); print(f"F1-Macro: {f1_macro:.4f}"); print(report_str)
    os.makedirs(os.path.dirname(output_cm_path), exist_ok=True)
    # Truyền full class names cho plot để trục được nhất quán
    plot_custom_confusion_matrix(y_true_final, y_pred_final, target_class_names_for_report, output_cm_path, labels_for_cm=range(len(target_class_names_for_report)))

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w') as f:
        f.write(f"Evaluation Report for: {model_full_name}\n")
        f.write(f"Confusion Matrix saved to: {os.path.basename(output_cm_path)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
        f.write(f"Macro F1-Score: {f1_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"Báo cáo chi tiết đã lưu vào: {output_report_path}")


def run_audio_lstm_evaluation_on_test_in_domain():
    print("===== ĐÁNH GIÁ AUDIO LSTM (FINE-TUNED TRÊN YOUTUBE) TRÊN TẬP TEST IN-DOMAIN =====")
    if hasattr(cfg, 'ensure_output_directories'): cfg.ensure_output_directories()

    # --- Tải các thành phần cần thiết ---
    print("\n--- Tải dữ liệu Test In-Domain Metadata, Encoders, Scaler, MaxLen và Mô hình Audio LSTM ---")
    try:
        test_id_metadata_df = pd.read_csv(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH)
        print(f"  Đã tải {len(test_id_metadata_df)} record từ metadata Test In-Domain: {cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH}")

        with open(cfg.FUSION_LABEL_ENCODER_PATH, 'rb') as f:
            fusion_label_encoder = pickle.load(f)
        print(f"  Đã tải Fusion LabelEncoder (dùng để đánh giá chung): {list(fusion_label_encoder.classes_)}")

        with open(cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH, 'rb') as f:
            audio_yt_native_label_encoder = pickle.load(f) # Encoder gốc của mô hình audio
        print(f"  Đã tải Audio YouTube Native LabelEncoder: {list(audio_yt_native_label_encoder.classes_)}")

        audio_lstm_model = load_model(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
        print(f"  Đã tải mô hình Audio LSTM fine-tuned từ: {cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH}")

        with open(cfg.AUDIO_YOUTUBE_SCALER_PATH, 'rb') as f:
            audio_scaler_yt = pickle.load(f)
        print(f"  Đã tải Audio YouTube Scaler từ: {cfg.AUDIO_YOUTUBE_SCALER_PATH}")

        with open(cfg.AUDIO_YOUTUBE_MAX_LEN_PATH, 'rb') as f:
            audio_max_len_yt = pickle.load(f)
        print(f"  Đã tải Audio YouTube MaxLen: {audio_max_len_yt}")

    except FileNotFoundError as e:
        print(f"LỖI: Thiếu file cần thiết để đánh giá: {e}")
        return
    except Exception as e:
        print(f"LỖI không xác định khi tải các thành phần: {e}")
        return

    y_true_for_audio_eval_encoded_list = []
    y_pred_for_audio_eval_encoded_list = []

    print(f"\n--- Thực hiện dự đoán trên {len(test_id_metadata_df)} segment của tập Test In-Domain ---")
    for index, row in tqdm(test_id_metadata_df.iterrows(), total=test_id_metadata_df.shape[0], desc="Evaluating Audio LSTM on In-Domain Test"):
        original_video_filename_no_ext = str(row['original_video']).strip()
        segment_id_from_metadata = str(row['segment_id']).strip() # Dùng để lấy segment index
        true_label_text_from_metadata = str(row['label']).strip()

        try:
            # Nhãn thực tế luôn được mã hóa bằng FUSION_LABEL_ENCODER để so sánh
            gt_encoded_fusion_space = fusion_label_encoder.transform([true_label_text_from_metadata])[0]
        except ValueError:
            # print(f"  CẢNH BÁO: Nhãn GT '{true_label_text_from_metadata}' cho seg '{segment_id_from_metadata}' không có trong Fusion LE. Bỏ qua.")
            continue

        # --- Xây dựng đường dẫn đến video gốc ---
        # Giả định 'true_label_text_from_metadata' là tên thư mục lớp trong YOUTUBE_RAW_VIDEOS_BASE_DIR
        original_video_folder = os.path.join(cfg.YOUTUBE_RAW_VIDEOS_BASE_DIR, true_label_text_from_metadata)
        possible_video_files = glob.glob(os.path.join(original_video_folder, f"{original_video_filename_no_ext}.*"))
        actual_original_video_path = None
        for f_path_check in possible_video_files:
             if f_path_check.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                 actual_original_video_path = f_path_check
                 break

        if not actual_original_video_path:
            # print(f"  CẢNH BÁO: Không tìm thấy video gốc cho '{original_video_filename_no_ext}'. Bỏ qua segment '{segment_id_from_metadata}'.")
            y_true_for_audio_eval_encoded_list.append(gt_encoded_fusion_space)
            try: y_pred_for_audio_eval_encoded_list.append(fusion_label_encoder.transform(["BACKGROUND"])[0])
            except: y_pred_for_audio_eval_encoded_list.append(-1)
            continue

        # --- Lấy đoạn audio gốc ---
        audio_segment_raw = get_audio_segment_from_video_for_evaluation(
            actual_original_video_path,
            segment_id_from_metadata, # Truyền segment_id để hàm tự parse index
            cfg.AUDIO_SEGMENT_DURATION,
            cfg.AUDIO_SAMPLE_RATE
        )

        # --- Dự đoán bằng Audio LSTM ---
        pred_encoded_fusion_space, _ = get_audio_model_prediction(
            audio_lstm_model,
            audio_segment_raw,
            audio_scaler_yt,
            audio_max_len_yt,
            audio_yt_native_label_encoder, # Encoder gốc của audio model
            fusion_label_encoder         # Encoder mục tiêu để đánh giá
        )

        y_true_for_audio_eval_encoded_list.append(gt_encoded_fusion_space)
        y_pred_for_audio_eval_encoded_list.append(pred_encoded_fusion_space) # đã là encoded theo fusion_le hoặc -1


    # --- Đánh giá và Báo cáo ---
    if not y_true_for_audio_eval_encoded_list :
        print("\nKhông có dữ liệu hợp lệ để đánh giá Audio LSTM trên Test In-Domain.")
        return

    # Chuyển sang NumPy array
    y_true_eval = np.array(y_true_for_audio_eval_encoded_list)
    y_pred_eval = np.array(y_pred_for_audio_eval_encoded_list)


    model_name_for_report = f"Audio_LSTM_FineTuned_YT_on_Fusion_In_Domain_Test" # Tên rõ ràng hơn
    output_cm_path = cfg.AUDIO_LSTM_TEST_IN_DOMAIN_CM_PLOT_PATH # Đường dẫn từ config
    output_report_path = os.path.join(cfg.FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN, 'report_audio_lstm_fine_tuned_yt_on_fusion_in_domain_test.txt')

    evaluate_and_report_single_model( # Đổi tên hàm cho nhất quán
        y_true_eval, # Truyền array
        y_pred_eval, # Truyền array
        model_name_for_report,
        list(fusion_label_encoder.classes_), # Luôn dùng FUSION_CLASS_NAMES cho report
        output_cm_path,
        output_report_path
    )
    print("=" * 60)

if __name__ == '__main__':
    dirs_to_ensure = [
        cfg.FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN,
        cfg.FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN,
        cfg.INTERIM_YOUTUBE_AUDIO_DIR
    ]
    for d_path in dirs_to_ensure:
        os.makedirs(d_path, exist_ok=True)

    run_audio_lstm_evaluation_on_test_in_domain()