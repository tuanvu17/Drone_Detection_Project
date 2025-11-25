# Drone_Detection_Project/src/debug/check_single_model_inputs.py
import os
import sys
import pickle
import cv2
import librosa
import numpy as np
import torch
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

# --- Thêm đường dẫn Project Root ---
PROJECT_ROOT_DEBUG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT_DEBUG not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DEBUG)
# --- Kết thúc thêm đường dẫn ---

try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.data_processing.prepare_fusion_finetune_data import preprocess_frame_for_yolo_input # Cần cho VCam
except ImportError as e:
    print(f"Lỗi import trong check_single_model_inputs.py: {e}")
    exit()

cfg = get_config()
device_torch_debug = torch.device("cpu")

def debug_vcam_yolo_prediction(yolo_model, image_path, target_imgsz, conf_thresh, yolo_class_names):
    print(f"\n--- DEBUG: VCam YOLO on: {os.path.basename(image_path)} ---")
    if not os.path.exists(image_path):
        print(f"  Ảnh không tồn tại: {image_path}"); return "ERROR_IMG_NOT_FOUND", 0.0
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  Không thể đọc ảnh: {image_path}"); return "ERROR_IMG_READ", 0.0

    print(f"  Chạy YOLO predict (imgsz={target_imgsz}, conf={conf_thresh})...")
    results = yolo_model.predict(source=frame.copy(), imgsz=target_imgsz, conf=conf_thresh, verbose=False, device=yolo_model.device)
    
    if results and results[0].boxes:
        print(f"  YOLO tìm thấy {len(results[0].boxes)} box(es).")
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = yolo_class_names[cls_id] if cls_id < len(yolo_class_names) else "UNKNOWN_ID"
            print(f"    Box {i}: Class='{cls_name}', Confidence={conf:.4f}, XYXY={box.xyxy[0].cpu().numpy()}")
        # Trả về lớp của box có confidence cao nhất (logic đơn giản)
        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        best_cls_id = int(best_box.cls[0])
        best_cls_name = yolo_class_names[best_cls_id] if best_cls_id < len(yolo_class_names) else "UNKNOWN_ID"
        return best_cls_name, float(best_box.conf[0])
    else:
        print("  YOLO không tìm thấy đối tượng nào với ngưỡng confidence này.")
        return "BACKGROUND_BY_YOLO_NO_DET", 0.0


def debug_audio_lstm_prediction(audio_model_keras, audio_segment_path, scaler_audio, max_len_audio, audio_model_label_encoder):
    print(f"\n--- DEBUG: Audio LSTM on: {os.path.basename(audio_segment_path)} ---")
    if not os.path.exists(audio_segment_path):
        print(f"  Audio segment không tồn tại: {audio_segment_path}"); return "ERROR_AUDIO_NOT_FOUND", 0.0
    try:
        audio_seg_data, sr_loaded = librosa.load(audio_segment_path, sr=cfg.AUDIO_SAMPLE_RATE)
        if sr_loaded != cfg.AUDIO_SAMPLE_RATE:
            print(f"  Cảnh báo SR: {sr_loaded} vs {cfg.AUDIO_SAMPLE_RATE}")
        if len(audio_seg_data) < int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.5):
            print("  Audio segment quá ngắn."); return "ERROR_AUDIO_TOO_SHORT", 0.0

        print("  Trích xuất MFCC...")
        mfcc_s = extract_mfcc(audio_seg_data, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
        print(f"    MFCC shape (raw): {mfcc_s.shape}")

        mfcc_p, len_after_pad = pad_features([mfcc_s], max_len=max_len_audio)
        if mfcc_p.size == 0: print("  MFCC rỗng sau padding."); return "ERROR_AUDIO_PAD", 0.0
        print(f"    MFCC shape (padded to {len_after_pad}): {mfcc_p.shape}")
        
        mfcc_sc, _ = scale_features(mfcc_p, scaler=scaler_audio)
        print(f"    MFCC shape (scaled): {mfcc_sc.shape}")
        print(f"    MFCC scaled min: {mfcc_sc.min()}, max: {mfcc_sc.max()}, mean: {mfcc_sc.mean()}")


        print("  Chạy Audio Keras predict...")
        audio_probs = audio_model_keras.predict(mfcc_sc, verbose=0)[0]
        audio_pred_idx = np.argmax(audio_probs)
        audio_pred_class_name = audio_model_label_encoder.classes_[audio_pred_idx]
        confidence = float(audio_probs[audio_pred_idx])
        print(f"    Xác suất các lớp (theo audio_model_label_encoder): {audio_model_label_encoder.classes_} -> {audio_probs}")
        return audio_pred_class_name, confidence
    except Exception as e_audio:
        print(f"  Lỗi khi xử lý audio segment: {e_audio}")
        import traceback
        traceback.print_exc()
        return "ERROR_AUDIO_PROCESSING", 0.0


if __name__ == "__main__":
    print("===== BẮT ĐẦU DEBUG INPUTS CHO CÁC MÔ HÌNH ĐƠN LẺ =====")
    if hasattr(cfg, 'ensure_output_directories'): cfg.ensure_output_directories()

    # --- Tải các thành phần cần thiết ---
    print("\n--- Tải các mô hình và thành phần tiền xử lý ---")
    try:
        vcam_yolo_detector_dbg = YOLO(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        vcam_yolo_detector_dbg.to(device_torch_debug)
        print(f"  Đã tải VCam YOLO Detector: {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH}")

        audio_model_dbg = load_model(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
        print(f"  Đã tải Audio LSTM Model: {cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH}")

        with open(cfg.AUDIO_YOUTUBE_SCALER_PATH, 'rb') as f: audio_scaler_dbg = pickle.load(f)
        with open(cfg.AUDIO_YOUTUBE_MAX_LEN_PATH, 'rb') as f: audio_max_len_dbg = pickle.load(f)
        with open(cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH, 'rb') as f: audio_le_dbg = pickle.load(f) # Encoder của audio model
        with open(cfg.FUSION_LABEL_ENCODER_PATH, 'rb') as f: fusion_le_dbg = pickle.load(f) # Encoder của fusion
        print("  Đã tải Scaler, MaxLen, LabelEncoders.")
    except Exception as e:
        print(f"LỖI khi tải thành phần: {e}"); exit()

    # --- Tải Metadata của Test In-Domain ---
    print(f"\n--- Tải Metadata Test In-Domain từ: {cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH} ---")
    if not os.path.exists(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH):
        print(f"LỖI: File metadata Test In-Domain không tồn tại. Hãy chạy fine_tune_fusion_pipeline.py trước."); exit()
    try:
        test_id_metadata_df = pd.read_csv(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH)
        print(f"  Đã tải {len(test_id_metadata_df)} segment metadata cho Test In-Domain.")
    except Exception as e:
        print(f"Lỗi khi đọc metadata Test In-Domain: {e}"); exit()

    # --- Chọn một vài mẫu để debug ---
    # Lấy các mẫu có nhãn DRONE và HELICOPTER từ Test In-Domain metadata
    drone_samples = test_id_metadata_df[test_id_metadata_df['label'] == 'DRONE'].head(2)
    helicopter_samples = test_id_metadata_df[test_id_metadata_df['label'] == 'HELICOPTER'].head(2)
    background_samples = test_id_metadata_df[test_id_metadata_df['label'] == 'BACKGROUND'].head(2) # Nếu có

    samples_to_debug = pd.concat([drone_samples, helicopter_samples, background_samples])

    if samples_to_debug.empty:
        print("\nKhông tìm thấy mẫu DRONE hoặc HELICOPTER nào trong metadata của Test In-Domain để debug.")
        print("Vui lòng kiểm tra file metadata hoặc thêm dữ liệu.")
    else:
        print(f"\nSẽ debug trên {len(samples_to_debug)} mẫu đã chọn...")

    for index, row in samples_to_debug.iterrows():
        print(f"\n================ DEBUGGING SEGMENT ID: {row.get('segment_id', index)} - Ground Truth: {row['label']} ================")
        
        # Đường dẫn đến file gốc (ảnh VCam và audio segment .wav)
        # FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR là thư mục chứa các file segment gốc
        # mà prepare_fusion_finetune_data.py đã lưu lại.
        vcam_frame_original_rel_path = row.get('vcam_frame_original_file')
        audio_segment_original_rel_path = row.get('audio_segment_original_file')

        if not vcam_frame_original_rel_path or not audio_segment_original_rel_path:
            print("  Lỗi: Thiếu đường dẫn đến file gốc trong metadata.")
            continue

        vcam_original_abs_path = os.path.join(cfg.FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR, vcam_frame_original_rel_path)
        audio_original_abs_path = os.path.join(cfg.FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR, audio_segment_original_rel_path)

        # 1. Kiểm tra VCam YOLO
        pred_vcam_name, conf_vcam = debug_vcam_yolo_prediction(
            vcam_yolo_detector_dbg,
            vcam_original_abs_path,
            cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE, # imgsz của VCam YOLO detector
            0.1, # Thử với ngưỡng confidence thấp hơn để xem có gì không
            vcam_yolo_detector_dbg.names # Lấy names từ model
        )
        print(f"  ==> VCam YOLO Predicted: Class='{pred_vcam_name}', Confidence={conf_vcam:.4f}")

        # 2. Kiểm tra Audio LSTM
        pred_audio_name, conf_audio = debug_audio_lstm_prediction(
            audio_model_dbg,
            audio_original_abs_path,
            audio_scaler_dbg, # Scaler từ audio YT fine-tune
            audio_max_len_dbg,  # MaxLen từ audio YT fine-tune
            audio_le_dbg        # LabelEncoder của audio YT fine-tune
        )
        print(f"  ==> Audio LSTM Predicted: Class='{pred_audio_name}', Confidence={conf_audio:.4f}")

    print("\n===== DEBUGGING FINISHED =====")