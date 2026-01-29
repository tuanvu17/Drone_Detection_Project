# Drone_Detection_Project/src/evaluation/compare_models_on_test_set.py
import os
# Đặt ở đây nếu muốn script này luôn chạy trên CPU khi được gọi
# Hoặc đặt ở file main_evaluate_comparison.py để kiểm soát tập trung
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import glob
import cv2
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from ultralytics import YOLO
import tensorflow as tf # Import sau khi có thể đã đặt CUDA_VISIBLE_DEVICES
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import từ các module trong src
try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
except ImportError:
    import sys
    current_dir_comp = os.path.dirname(os.path.abspath(__file__))
    src_dir_comp = os.path.dirname(current_dir_comp)
    project_root_comp = os.path.dirname(src_dir_comp)
    if project_root_comp not in sys.path: sys.path.insert(0, project_root_comp)
    if src_dir_comp not in sys.path: sys.path.insert(0, src_dir_comp)
    from config_loader.loader import get_config
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input
    from evaluation.plotting_utils import plot_custom_confusion_matrix

cfg_comp = get_config() # Tải config cho module này

def get_yolo_segment_prediction(yolo_model, frame, target_imgsz, conf_thresh, yolo_class_names_from_model, target_fusion_classes):
    """
    Chạy YOLO trên frame, lấy đối tượng có confidence cao nhất và trả về tên lớp (trong target_fusion_classes).
    Nếu không có phát hiện hoặc lớp không khớp, trả về BACKGROUND.
    """
    # Đảm bảo model YOLO chạy trên CPU nếu device_torch là 'cpu'
    # device_to_use = yolo_model.device # Lấy device từ model đã load
    # Tuy nhiên, để chắc chắn, có thể truyền device vào
    predictions = yolo_model.predict(source=frame, imgsz=target_imgsz, conf=conf_thresh, verbose=False) # Bỏ device, YOLO tự xử lý
    
    best_class_name = "BACKGROUND" # Mặc định
    max_conf = 0.0

    if predictions and predictions[0].boxes:
        for box in predictions[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name_yolo = yolo_class_names_from_model[cls_id] # Sử dụng names từ model

            if cls_name_yolo in target_fusion_classes and conf > max_conf :
                if best_class_name == "BACKGROUND" or cls_name_yolo != "BACKGROUND":
                    max_conf = conf
                    best_class_name = cls_name_yolo
                elif cls_name_yolo == "BACKGROUND" and best_class_name == "BACKGROUND" and conf > max_conf:
                     max_conf = conf
                     best_class_name = cls_name_yolo
    return best_class_name

def run_model_comparison_on_evaluation_set(device_for_pytorch_models='cpu'): # Thêm tham số device
    print("===== STARTING MODEL COMPARISON ON FINAL EVALUATION TEST SET =====")
    if hasattr(cfg_comp, 'ensure_output_directories'): cfg_comp.ensure_output_directories()

    print(f"Đang tải Ground Truth Metadata từ: {cfg_comp.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE}")
    if not os.path.exists(cfg_comp.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE):
        print(f"LỖI: File metadata ground truth không tồn tại. Vui lòng chạy 'prepare_evaluation_test_data.py' trước.")
        return
    try:
        import pandas as pd
        gt_metadata_df = pd.read_csv(cfg_comp.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE)
        print(f"Đã tải {len(gt_metadata_df)} segment ground truth.")
    except Exception as e:
        print(f"Lỗi khi đọc file metadata ground truth: {e}"); return

    eval_label_encoder = LabelEncoder()
    eval_label_encoder.fit(cfg_comp.MASTER_CLASS_LIST_FUSION)
    ground_truth_encoded = eval_label_encoder.transform(gt_metadata_df['ground_truth_label'])
    class_names_for_report = list(eval_label_encoder.classes_)
    
    # Lưu GT và Class Names
    # (Đảm bảo thư mục EVALUATION_REPORTS_METRICS_DIR tồn tại)
    os.makedirs(cfg_comp.EVALUATION_REPORTS_METRICS_DIR, exist_ok=True)
    with open(os.path.join(cfg_comp.EVALUATION_REPORTS_METRICS_DIR, 'eval_ground_truth_encoded.pkl'), 'wb') as f: pickle.dump(ground_truth_encoded, f)
    with open(os.path.join(cfg_comp.EVALUATION_REPORTS_METRICS_DIR, 'eval_class_names.pkl'), 'wb') as f: pickle.dump(class_names_for_report, f)

    print("\n--- Đang tải các mô hình và thành phần tiền xử lý ---")
    try:
        # Thiết bị cho PyTorch models
        pt_device = torch.device(device_for_pytorch_models)
        print(f"  Thiết bị PyTorch sẽ sử dụng: {pt_device}")

        vcam_yolo_detector = YOLO(cfg_comp.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        vcam_yolo_detector.to(pt_device) # Chuyển model YOLO sang device
        print(f"  Đã tải VCam YOLO Detector: {cfg_comp.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH} trên {vcam_yolo_detector.device}")

        vcam_yolo_for_feat_ext = YOLO(cfg_comp.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION)
        vcam_feat_extractor = FeatureExtractorVCam(vcam_yolo_for_feat_ext.model.model, extraction_layer_index=8)
        vcam_feat_extractor.to(pt_device).eval() # Chuyển feature extractor sang device
        print(f"  Đã tải VCam Feature Extractor (từ {cfg_comp.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION}) trên {pt_device}.")

        audio_yt_model = tf.keras.models.load_model(cfg_comp.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
        print(f"  Đã tải Audio YouTube Fine-tuned Model: {cfg_comp.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH}")
        with open(cfg_comp.AUDIO_YOUTUBE_SCALER_PATH, 'rb') as f: audio_yt_scaler = pickle.load(f)
        with open(cfg_comp.AUDIO_YOUTUBE_MAX_LEN_PATH, 'rb') as f: audio_yt_max_len = pickle.load(f)
        print("  Đã tải Scaler và MaxLen cho Audio YouTube Fine-tuned.")

        fusion_model = tf.keras.models.load_model(cfg_comp.BEST_FUSION_MODEL_SAVE_PATH)
        print(f"  Đã tải Fusion Model: {cfg_comp.BEST_FUSION_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"LỖI khi tải một trong các mô hình hoặc thành phần: {e}"); import traceback; traceback.print_exc(); return

    vcam_yolo_preds_encoded = []
    audio_lstm_preds_encoded = []
    fusion_model_preds_encoded = []

    print("\n--- Thực hiện dự đoán trên tập test đánh giá ---")
    for index, row in tqdm(gt_metadata_df.iterrows(), total=gt_metadata_df.shape[0], desc="Predicting on Eval Test Set"):
        vcam_frame_path = os.path.join(cfg_comp.EVALUATION_TEST_SEGMENTS_BASE_DIR, row['path_to_vcam_frame'])
        audio_segment_path = os.path.join(cfg_comp.EVALUATION_TEST_SEGMENTS_BASE_DIR, row['path_to_audio_segment'])
        frame_cv = None # Khởi tạo để tránh lỗi nếu đọc file VCam thất bại

        # a. Dự đoán bằng VCam YOLO
        try:
            if not os.path.exists(vcam_frame_path): raise FileNotFoundError(f"Ảnh VCam không tồn tại: {vcam_frame_path}")
            frame_cv = cv2.imread(vcam_frame_path)
            if frame_cv is None: raise ValueError(f"Không thể đọc frame VCam: {vcam_frame_path}")
            
            vcam_pred_class_name = get_yolo_segment_prediction(
                vcam_yolo_detector, frame_cv, cfg_comp.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                0.25, vcam_yolo_detector.names, class_names_for_report
            )
            vcam_yolo_preds_encoded.append(eval_label_encoder.transform([vcam_pred_class_name])[0])
        except Exception as e_vcam:
            print(f"Lỗi VCam YOLO predict cho {row.get('segment_id', index)}: {e_vcam}. Gán BACKGROUND.")
            vcam_yolo_preds_encoded.append(eval_label_encoder.transform(["BACKGROUND"])[0])

        # b. Dự đoán bằng Audio LSTM
        try:
            if not os.path.exists(audio_segment_path): raise FileNotFoundError(f"File audio không tồn tại: {audio_segment_path}")
            audio_seg_data, _ = librosa.load(audio_segment_path, sr=cfg_comp.AUDIO_SAMPLE_RATE)
            mfcc_s = extract_mfcc(audio_seg_data, cfg_comp.AUDIO_SAMPLE_RATE, cfg_comp.AUDIO_N_MFCC, cfg_comp.AUDIO_N_FFT, cfg_comp.AUDIO_HOP_LENGTH)
            mfcc_p, _ = pad_features([mfcc_s], max_len=audio_yt_max_len)
            if mfcc_p.size == 0: raise ValueError("MFCC rỗng sau padding cho audio")
            mfcc_sc, _ = scale_features(mfcc_p, scaler=audio_yt_scaler)
            
            audio_probs = audio_yt_model.predict(mfcc_sc, verbose=0)[0]
            audio_pred_idx = np.argmax(audio_probs)
            audio_lstm_preds_encoded.append(audio_pred_idx)
        except Exception as e_audio:
            print(f"Lỗi Audio LSTM predict cho {row.get('segment_id', index)}: {e_audio}. Gán BACKGROUND.")
            audio_lstm_preds_encoded.append(eval_label_encoder.transform(["BACKGROUND"])[0])

        # c. Dự đoán bằng Fusion Model
        try:
            # Trích xuất VCam feature cho fusion
            if frame_cv is None: # Nếu frame đã lỗi từ bước VCam YOLO
                vcam_feat_for_fusion_np = np.zeros((1, cfg_comp.FUSION_VCAM_FEATURE_DIM), dtype=np.float32)
            else:
                vcam_tensor = preprocess_frame_for_yolo_input(frame_cv, cfg_comp.VCAM_YOLO_IMG_SIZE) # Dùng imgsz của model đã tạo feature extractor
                with torch.no_grad():
                    vcam_feat_for_fusion_np = vcam_feat_extractor(vcam_tensor.to(pt_device)).cpu().numpy()

            # MFCC đã có từ bước b (mfcc_sc)
            fusion_probs = fusion_model.predict([mfcc_sc, vcam_feat_for_fusion_np], verbose=0)[0]
            fusion_pred_idx = np.argmax(fusion_probs)
            fusion_model_preds_encoded.append(fusion_pred_idx)
        except Exception as e_fusion:
            print(f"Lỗi Fusion predict cho {row.get('segment_id', index)}: {e_fusion}. Gán BACKGROUND.")
            fusion_model_preds_encoded.append(eval_label_encoder.transform(["BACKGROUND"])[0])

    # --- 5. Lưu trữ và Trình bày Kết quả So sánh ---
    # ... (Giữ nguyên phần này như code bạn đã cung cấp ở câu hỏi 34, chỉ cần đảm bảo
    #      cfg_comp được sử dụng thay cho cfg nếu bạn đã đổi tên biến config ở đầu file này)
    #      Ví dụ: cfg.EVALUATION_REPORTS_FIGURES_DIR -> cfg_comp.EVALUATION_REPORTS_FIGURES_DIR
    # ...
    print("\n--- KẾT QUẢ ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH ---")
    report_outputs = {}
    models_to_evaluate = {
        "VCam_YOLO_YT_FT": np.array(vcam_yolo_preds_encoded), # Đổi tên để rõ hơn
        "Audio_LSTM_YT_FT": np.array(audio_lstm_preds_encoded),
        "Early_Fusion_VCam_Audio": np.array(fusion_model_preds_encoded)
    }
    # ... (Phần tính toán report, vẽ CM riêng, nối report, vẽ CM chung, lưu pkl) ...
    # (Copy từ phiên bản hoạt động trước đó của bạn, đảm bảo dùng cfg_comp)
    full_classification_report_str = ""
    for model_name, preds_encoded in models_to_evaluate.items():
        print(f"\n--- Kết quả cho Mô hình: {model_name} ---")
        if len(preds_encoded) != len(ground_truth_encoded):
            print(f"  CẢNH BÁO: Số lượng dự đoán ({len(preds_encoded)}) không khớp với ground truth ({len(ground_truth_encoded)}).")
            continue
        report = classification_report(ground_truth_encoded, preds_encoded,
                                       target_names=class_names_for_report,
                                       labels=range(len(class_names_for_report)),
                                       zero_division=0)
        print(report)
        report_outputs[model_name] = report
        cm_filename = os.path.join(cfg_comp.EVALUATION_REPORTS_FIGURES_DIR, f"cm_{model_name.lower()}_eval_test.png")
        plot_custom_confusion_matrix(ground_truth_encoded, preds_encoded, class_names_for_report, filename=cm_filename)
        full_classification_report_str += f"===== {model_name} =====\n{report}\n\n"

    report_file_path = os.path.join(cfg_comp.EVALUATION_REPORTS_METRICS_DIR, "all_models_classification_reports_eval_test.txt")
    os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
    with open(report_file_path, 'w') as f: f.write(full_classification_report_str)
    print(f"\nBáo cáo phân loại đã lưu vào: {report_file_path}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Confusion matrices on OOD dataset: (a) VCam, (b) Audio, (c) Early Fusion', fontsize=16)
    idx = 0
    for model_name, preds_encoded in models_to_evaluate.items():
        if len(preds_encoded) != len(ground_truth_encoded): continue
        ax = axes[idx]
        cm = sklearn_confusion_matrix(ground_truth_encoded, preds_encoded, labels=range(len(class_names_for_report)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names_for_report, yticklabels=class_names_for_report, ax=ax, cbar=(idx == 2))
        ax.set_title(model_name)
        ax.set_xlabel('Predicted Label')
        if idx == 0: ax.set_ylabel('True Label')
        else: ax.set_ylabel('')
        idx += 1
    for i in range(idx, 3): fig.delaxes(axes[i])
    comparison_cm_path = os.path.join(cfg_comp.EVALUATION_REPORTS_FIGURES_DIR, "cm_comparison_all_models_eval_test.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(comparison_cm_path)
    print(f"Ảnh so sánh CM đã lưu vào: {comparison_cm_path}")
    plt.close(fig)

    # Lưu các mảng dự đoán
    for model_name, preds_encoded in models_to_evaluate.items():
        pred_save_path = os.path.join(cfg_comp.EVALUATION_REPORTS_METRICS_DIR, f'eval_preds_{model_name.lower()}.pkl')
        with open(pred_save_path, 'wb') as f: pickle.dump(preds_encoded, f)
        print(f"Đã lưu dự đoán của {model_name} vào: {pred_save_path}")

    print("===== MODEL COMPARISON ON FINAL EVALUATION TEST SET FINISHED =====")

# Không cần if __name__ == '__main__': ở đây nữa vì file này sẽ được gọi từ main_evaluate_comparison.py