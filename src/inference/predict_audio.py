# Drone_Detection_Project/src/inference/predict_audio.py
import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob # Thêm glob để tìm file

# Import các hàm tiện ích tiền xử lý từ data_processing
# và config_loader
try:
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root_from_here = os.path.dirname(src_dir)
    if project_root_from_here not in sys.path:
        sys.path.insert(0, project_root_from_here)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from config_loader.loader import get_config


def classify_audio_files(audio_file_paths,
                         model_path,
                         label_encoder_path,
                         scaler_path,
                         max_len_path,
                         segment_duration, sr, n_mfcc, n_fft, hop_length):
    """
    Phân loại một danh sách các file âm thanh sử dụng mô hình đã huấn luyện.
    (Nội dung hàm này giữ nguyên như trong câu trả lời trước)
    """
    results = []
    try:
        print(f"Đang tải mô hình từ: {model_path}")
        model = load_model(model_path)
        print(f"Đang tải LabelEncoder từ: {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Đang tải Scaler từ: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Đang tải max_len từ: {max_len_path}")
        with open(max_len_path, 'rb') as f:
            max_len = pickle.load(f)
        print("Tải mô hình và các thành phần tiền xử lý thành công.")
        actual_classes = list(label_encoder.classes_)
    except Exception as e:
        print(f"LỖI: Không thể tải mô hình hoặc các thành phần tiền xử lý: {e}")
        for file_path in audio_file_paths:
            results.append({
                'file_path': file_path,
                'error': f"Không thể tải mô hình hoặc các thành phần tiền xử lý: {e}"
            })
        return results

    for file_path in audio_file_paths:
        print(f"\nĐang xử lý file: {file_path}")
        if not os.path.exists(file_path):
            print(f"  CẢNH BÁO: File không tồn tại: {file_path}. Bỏ qua.")
            results.append({'file_path': file_path, 'error': 'File not found'})
            continue
        try:
            audio, current_sr_load = librosa.load(file_path, sr=sr)
            if current_sr_load != sr:
                print(f"    Cảnh báo: File {file_path} được tải với tần số mẫu {current_sr_load}, nhưng mô hình mong đợi {sr}. Kết quả có thể bị ảnh hưởng.")

            segment_samples = int(segment_duration * sr)
            segment_features_list = []

            for start_sample in range(0, len(audio) - segment_samples + 1, segment_samples):
                segment = audio[start_sample : start_sample + segment_samples]
                if len(segment) == segment_samples:
                    mfccs = extract_mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    segment_features_list.append(mfccs)

            if not segment_features_list:
                print(f"  CẢNH BÁO: Không trích xuất được đoạn hợp lệ nào từ {file_path}. Bỏ qua.")
                results.append({'file_path': file_path, 'error': 'No valid segments extracted'})
                continue

            features_padded, _ = pad_features(segment_features_list, max_len=max_len)
            if features_padded.size == 0:
                 print(f"  CẢNH BÁO: Không có features hợp lệ sau khi đệm cho {file_path}. Bỏ qua.")
                 results.append({'file_path': file_path, 'error': 'No valid features after padding'})
                 continue

            features_scaled, _ = scale_features(features_padded, scaler=scaler)
            segment_pred_probs = model.predict(features_scaled, verbose=0)
            avg_probs = np.mean(segment_pred_probs, axis=0)
            pred_label_encoded = np.argmax(avg_probs)
            pred_label_name = label_encoder.inverse_transform([pred_label_encoded])[0]
            confidence = float(avg_probs[pred_label_encoded])
            probabilities = avg_probs.tolist()

            print(f"  Dự đoán: {pred_label_name} (Độ tự tin: {confidence:.4f})")
            results.append({
                'file_path': file_path,
                'predicted_class': pred_label_name,
                'confidence': confidence,
                'probabilities': probabilities,
                'all_class_names': actual_classes
            })
        except Exception as e:
            print(f"  LỖI khi xử lý file {file_path}: {e}")
            results.append({'file_path': file_path, 'error': str(e)})
    return results


if __name__ == "__main__":
    print("Bắt đầu kiểm thử hàm classify_audio_files...")

    # --- Lấy cấu hình ---
    try:
        cfg = get_config()
        # Đảm bảo các thư mục output tồn tại nếu hàm ensure_output_directories có trong cfg
        if hasattr(cfg, 'ensure_output_directories'):
            cfg.ensure_output_directories()
    except Exception as e:
        print(f"Lỗi khi tải hoặc thiết lập cấu hình: {e}.")
        print("Sẽ cố gắng sử dụng đường dẫn mặc định nếu có thể.")
        cfg = None # Đặt cfg là None để kiểm tra sau

    # --- Xác định đường dẫn đến thư mục chứa file âm thanh cần dự đoán ---
    # Sử dụng đường dẫn tuyệt đối bạn cung cấp
    # Nếu cfg không tải được, chúng ta cần một cách khác để lấy PROJECT_ROOT
    if cfg and hasattr(cfg, 'PROJECT_ROOT'):
        project_root_dir = cfg.PROJECT_ROOT
    else:
        # Tính toán project_root từ vị trí file hiện tại nếu cfg không có
        current_script_dir = os.path.dirname(os.path.abspath(__file__)) # src/inference
        src_dir_local = os.path.dirname(current_script_dir) # src
        project_root_dir = os.path.dirname(src_dir_local) # Drone_Detection_Project
        print(f"Không tải được cfg.PROJECT_ROOT, sử dụng project_root được tính toán: {project_root_dir}")


    # Đường dẫn tuyệt đối đến thư mục AUDIO2PREDICT
    # THAY ĐỔI DÒNG NÀY NẾU CẦN (nhưng nó nên lấy từ cấu trúc bạn cung cấp)
    audio_to_predict_dir = os.path.join(project_root_dir, 'data', 'raw', 'Audio_Original', 'AUDIO2PREDICT')
    print(f"Đang tìm file .wav trong thư mục: {audio_to_predict_dir}")

    if not os.path.isdir(audio_to_predict_dir):
        print(f"LỖI: Thư mục '{audio_to_predict_dir}' không tồn tại. Vui lòng tạo thư mục và đặt file audio vào đó.")
        files_to_classify = []
    else:
        # Tìm tất cả các file .wav trong thư mục đó
        files_to_classify = glob.glob(os.path.join(audio_to_predict_dir, "*.wav"))
        if not files_to_classify:
            print(f"Không tìm thấy file .wav nào trong: {audio_to_predict_dir}")
        else:
            print(f"Tìm thấy {len(files_to_classify)} file âm thanh để phân loại.")

    # --- Lấy các đường dẫn và tham số từ config (hoặc dùng giá trị mặc định nếu cfg lỗi) ---
    # Điều này rất quan trọng để hàm classify_audio_files nhận đúng các thành phần đã huấn luyện
    model_p = cfg.AUDIO_MODEL_SAVE_PATH if cfg and hasattr(cfg, 'AUDIO_MODEL_SAVE_PATH') else os.path.join(project_root_dir, 'models', 'audio_classifier', 'best_audio_model.keras')
    label_encoder_p = cfg.AUDIO_LABEL_ENCODER_PATH if cfg and hasattr(cfg, 'AUDIO_LABEL_ENCODER_PATH') else os.path.join(project_root_dir, 'models', 'audio_classifier', 'label_encoder_audio.pkl')
    scaler_p = cfg.AUDIO_SCALER_PATH if cfg and hasattr(cfg, 'AUDIO_SCALER_PATH') else os.path.join(project_root_dir, 'models', 'audio_classifier', 'scaler_audio.pkl')
    max_len_p = cfg.AUDIO_MAX_LEN_PATH if cfg and hasattr(cfg, 'AUDIO_MAX_LEN_PATH') else os.path.join(project_root_dir, 'models', 'audio_classifier', 'max_len_audio.pkl')

    seg_dur = cfg.AUDIO_SEGMENT_DURATION if cfg and hasattr(cfg, 'AUDIO_SEGMENT_DURATION') else 1.0
    sample_r = cfg.AUDIO_SAMPLE_RATE if cfg and hasattr(cfg, 'AUDIO_SAMPLE_RATE') else 44100
    n_mfcc_val = cfg.AUDIO_N_MFCC if cfg and hasattr(cfg, 'AUDIO_N_MFCC') else 13
    n_fft_val = cfg.AUDIO_N_FFT if cfg and hasattr(cfg, 'AUDIO_N_FFT') else int(sample_r * 0.025)
    hop_len_val = cfg.AUDIO_HOP_LENGTH if cfg and hasattr(cfg, 'AUDIO_HOP_LENGTH') else int(sample_r * 0.010)


    # Kiểm tra sự tồn tại của các file model cần thiết
    required_model_files = [model_p, label_encoder_p, scaler_p, max_len_p]
    missing_files = [f for f in required_model_files if not os.path.exists(f)]

    if missing_files:
        print("\nLỖI: Thiếu một hoặc nhiều file model/tiền xử lý cần thiết:")
        for f_path in missing_files:
            print(f"  - {f_path}")
        print("Vui lòng đảm bảo đã huấn luyện mô hình và các file này tồn tại.")
    elif not files_to_classify:
        print("Không có file âm thanh nào trong thư mục AUDIO2PREDICT để phân loại.")
    else:
        # Gọi hàm phân loại
        predictions = classify_audio_files(
            audio_file_paths=files_to_classify,
            model_path=model_p,
            label_encoder_path=label_encoder_p,
            scaler_path=scaler_p,
            max_len_path=max_len_p,
            segment_duration=seg_dur,
            sr=sample_r,
            n_mfcc=n_mfcc_val,
            n_fft=n_fft_val,
            hop_length=hop_len_val
        )

        print("\n--- Kết quả Phân loại ---")
        for result in predictions:
            if 'error' in result:
                print(f"File: {result['file_path']} - Lỗi: {result['error']}")
            else:
                print(f"File: {os.path.basename(result['file_path'])}") # In tên file ngắn gọn
                print(f"  Đường dẫn đầy đủ: {result['file_path']}")
                print(f"  Lớp dự đoán: {result['predicted_class']}")
                print(f"  Độ tự tin: {result['confidence']:.4f}")
                # Hiển thị xác suất cho từng lớp một cách rõ ràng hơn
                print(f"  Xác suất các lớp:")
                for class_name, prob in zip(result['all_class_names'], result['probabilities']):
                    print(f"    - {class_name}: {prob:.4f}")