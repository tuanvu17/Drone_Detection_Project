# Drone_Detection_Project/src/evaluation/evaluate_models_on_in_domain.py
import os
# Đặt os.environ ở đây nếu muốn script này luôn chạy trên CPU khi được gọi từ bất cứ đâu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Buộc TF chạy trên CPU

import glob
import cv2
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from ultralytics import YOLO
import tensorflow as tf # Import sau khi đã đặt CUDA_VISIBLE_DEVICES
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    from src.config_loader.loader import get_config
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
    from src.data_processing.prepare_fusion_finetune_data import preprocess_frame_for_yolo_input
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
    from data_processing.prepare_fusion_finetune_data import preprocess_frame_for_yolo_input
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features

cfg = get_config()
# device_torch_eval = torch.device("cpu") # YOLO sẽ tự xử lý device hoặc nhận từ predict/val

def get_yolo_prediction_for_frame(yolo_model, image_path, target_imgsz, conf_thresh,
                                  yolo_class_names_from_model,
                                  target_fusion_label_encoder,
                                  default_class_fusion_space="BACKGROUND"):
    if not os.path.exists(image_path):
        try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0] , 0.0
        except: return -1, 0.0 # -1 nếu BACKGROUND không có trong target_fusion_label_encoder
    frame = cv2.imread(image_path)
    if frame is None:
        try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
        except: return -1, 0.0

    # Sử dụng device đã được set cho yolo_model khi tải, hoặc truyền vào đây
    device_to_use_yolo = yolo_model.device if hasattr(yolo_model, 'device') else 'cpu'
    predictions = yolo_model.predict(source=frame, imgsz=target_imgsz, conf=conf_thresh, verbose=False, device=device_to_use_yolo)
    
    best_class_name_yolo_mapped = default_class_fusion_space
    max_confidence = 0.0

    if predictions and predictions[0].boxes and hasattr(predictions[0].boxes, 'conf') and len(predictions[0].boxes.conf) > 0:
        for box_idx in range(len(predictions[0].boxes.conf)): # Lặp qua index
            conf = float(predictions[0].boxes.conf[box_idx])
            cls_id = int(predictions[0].boxes.cls[box_idx])
            if cls_id < len(yolo_class_names_from_model):
                cls_name_yolo_orig = yolo_class_names_from_model[cls_id]
                if conf > max_confidence:
                    max_confidence = conf
                    if cls_name_yolo_orig in target_fusion_label_encoder.classes_: # SỬA Ở ĐÂY
                        best_class_name_yolo_mapped = cls_name_yolo_orig
    try:
        return target_fusion_label_encoder.transform([best_class_name_yolo_mapped])[0], max_confidence
    except:
        try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
        except: return -1, 0.0


def get_audio_prediction_for_segment(audio_model_keras, audio_segment_path,
                                     scaler_audio, max_len_audio,
                                     audio_model_label_encoder, # Encoder của chính audio_model
                                     target_fusion_label_encoder, # Encoder của Fusion để ánh xạ
                                     default_class_fusion_space="BACKGROUND"):
    if not os.path.exists(audio_segment_path):
        try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
        except: return -1, 0.0
    try:
        audio_seg_data, _ = librosa.load(audio_segment_path, sr=cfg.AUDIO_SAMPLE_RATE)
        if len(audio_seg_data) < int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.5):
            try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
            except: return -1, 0.0

        mfcc_s = extract_mfcc(audio_seg_data, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
        mfcc_p_list, _ = pad_features([mfcc_s], max_len=max_len_audio) # pad_features trả về list (numpy array)
        if mfcc_p_list.size == 0:
             try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
             except: return -1, 0.0
        mfcc_p = mfcc_p_list # mfcc_p giờ là numpy array
        mfcc_sc, _ = scale_features(mfcc_p, scaler=scaler_audio)
        
        audio_probs = audio_model_keras.predict(mfcc_sc, verbose=0)[0]
        audio_pred_idx_in_audio_space = np.argmax(audio_probs)
        audio_pred_class_name = audio_model_label_encoder.classes_[audio_pred_idx_in_audio_space] # SỬA Ở ĐÂY
        confidence = float(audio_probs[audio_pred_idx_in_audio_space])

        if audio_pred_class_name in target_fusion_label_encoder.classes_: # SỬA Ở ĐÂY
            return target_fusion_label_encoder.transform([audio_pred_class_name])[0], confidence
        else:
            try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
            except: return -1, 0.0
    except Exception as e_audio:
        try: return target_fusion_label_encoder.transform([default_class_fusion_space])[0], 0.0
        except: return -1, 0.0

def evaluate_and_report_single_model(y_true_encoded, y_pred_encoded,
                                     model_full_name,
                                     target_class_names_for_report,
                                     output_cm_path,
                                     output_report_path):
    print(f"\n--- Đánh giá Mô hình: {model_full_name} ---")
    if len(y_true_encoded) == 0 or len(y_pred_encoded) == 0 or len(y_true_encoded) != len(y_pred_encoded):
        print(f"  CẢNH BÁO: Dữ liệu không hợp lệ cho {model_full_name}. GT: {len(y_true_encoded)}, Pred: {len(y_pred_encoded)}")
        return
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
    report_str = classification_report(y_true_encoded, y_pred_encoded,
                                       target_names=target_class_names_for_report,
                                       labels=list(range(len(target_class_names_for_report))), # Đảm bảo labels là list of int
                                       zero_division=0)
    print(f"Accuracy: {accuracy:.4f}"); print(f"F1-Weighted: {f1_weighted:.4f}"); print(f"F1-Macro: {f1_macro:.4f}"); print(report_str)
    os.makedirs(os.path.dirname(output_cm_path), exist_ok=True)
    plot_custom_confusion_matrix(y_true_encoded, y_pred_encoded, target_class_names_for_report, output_cm_path, title=f"CM - {model_full_name}") # Thêm title
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w') as f:
        f.write(f"Report: {model_full_name}\nCM: {os.path.basename(output_cm_path)}\nAcc: {accuracy:.4f}\nF1W: {f1_weighted:.4f}\nF1M: {f1_macro:.4f}\n\n{report_str}")
    print(f"Báo cáo đã lưu: {output_report_path}")


def combine_cms_horizontally(cm_paths_dict, output_combined_path):
    images = []; titles = []
    for title, p in cm_paths_dict.items():
        if os.path.exists(p): img = cv2.imread(p); images.append(img); titles.append(title)
    if not images: print("Không có ảnh CM để gộp."); return
    
    # Tìm chiều cao lớn nhất để resize các ảnh khác theo
    # hoặc chọn một chiều cao cố định
    if not images: return
    target_height = images[0].shape[0] if images else 600
    resized_images = []

    for i, img in enumerate(images):
        if img is None: print(f"Cảnh báo: Không thể đọc ảnh CM từ {cm_paths_dict.get(titles[i] if i < len(titles) else 'N/A')}"); continue
        h, w = img.shape[:2]; scale = target_height / h; new_w = int(w * scale)
        resized_img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
        # Không vẽ title lên ảnh ở đây nữa, sẽ dùng fig.suptitle
        resized_images.append(resized_img)
        
    if not resized_images: print("Không có ảnh CM hợp lệ sau resize."); return
    try:
        combined_image = cv2.hconcat(resized_images)
        os.makedirs(os.path.dirname(output_combined_path), exist_ok=True)
        cv2.imwrite(output_combined_path, combined_image)
        print(f"Ảnh gộp CM đã lưu: {output_combined_path}")
    except Exception as e: print(f"Lỗi gộp CM: {e}")


def run_evaluation_on_dataset_files(
        dataset_name_suffix,
        X_audio_test_features_for_fusion_path,
        X_vcam_test_features_for_fusion_path,
        y_test_true_labels_encoded_fusion_path,
        test_segments_metadata_path,
        raw_segments_storage_base_dir,
        audio_model_path,
        vcam_yolo_detector_path,
        fusion_model_path,
        audio_youtube_scaler_path,
        audio_youtube_max_len_path,
        audio_youtube_label_encoder_path,
        fusion_label_encoder_path,
        base_report_dir_figures_arg, # Đổi tên để tránh trùng với biến toàn cục
        base_report_dir_metrics_arg  # Đổi tên
    ):
    print(f"\n===== BẮT ĐẦU ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP: Test {dataset_name_suffix} =====")
    # Sử dụng tham số truyền vào
    os.makedirs(base_report_dir_figures_arg, exist_ok=True)
    os.makedirs(base_report_dir_metrics_arg, exist_ok=True)

    print("\n--- Tải dữ liệu Test, Encoders và Models ---")
    try:
        with open(X_audio_test_features_for_fusion_path, 'rb') as f: X_audio_for_fusion = pickle.load(f)
        with open(X_vcam_test_features_for_fusion_path, 'rb') as f: X_vcam_for_fusion = pickle.load(f)
        with open(y_test_true_labels_encoded_fusion_path, 'rb') as f: y_true_fusion_space_test = pickle.load(f) # Đổi tên biến
        with open(fusion_label_encoder_path, 'rb') as f: fusion_le = pickle.load(f)
        with open(audio_youtube_label_encoder_path, 'rb') as f: audio_le_yt = pickle.load(f)
        with open(audio_youtube_scaler_path, 'rb') as f: audio_scaler_yt = pickle.load(f)
        with open(audio_youtube_max_len_path, 'rb') as f: audio_max_len_yt = pickle.load(f)
        gt_metadata_df_test = pd.read_csv(test_segments_metadata_path) # Đổi tên biến
        if len(y_true_fusion_space_test) != len(gt_metadata_df_test): print("CẢNH BÁO: Số lượng nhãn test features không khớp metadata.")
        
        audio_model = load_model(audio_model_path)
        vcam_yolo_detector_model = YOLO(vcam_yolo_detector_path) # Đổi tên biến
        # vcam_yolo_detector_model.to(device_torch_eval) # Sẽ truyền device vào predict
        fusion_model = load_model(fusion_model_path)
        print("  Tải thành công dữ liệu, encoders và models.")
    except Exception as e: print(f"LỖI khi tải: {e}"); return

    preds_vcam_yolo_encoded_list = []
    preds_audio_lstm_encoded_list = []
    preds_fusion_encoded_list = []
    actual_gt_labels_for_eval_list = [] # Ground truth đã được mã hóa theo fusion_le

    print(f"\n--- Thực hiện dự đoán trên {len(gt_metadata_df_test)} segment của tập Test {dataset_name_suffix} ---")
    for index, row in tqdm(gt_metadata_df_test.iterrows(), total=gt_metadata_df_test.shape[0]):
        gt_label_text_current = str(row['label']).strip() # Lấy nhãn từ metadata gốc
        try:
            # Mã hóa GT của segment hiện tại bằng FUSION_LABEL_ENCODER
            gt_encoded_current_fusion_space = fusion_le.transform([gt_label_text_current])[0]
        except ValueError:
            # print(f"Cảnh báo: GT label '{gt_label_text_current}' không có trong fusion_le. Bỏ qua segment {index}.")
            continue # Bỏ qua segment này nếu GT không hợp lệ cho fusion space

        vcam_frame_rel = row.get('vcam_frame_original_file')
        vcam_pred_enc, _ = get_yolo_prediction_for_frame(
            vcam_yolo_detector_model,
            os.path.join(raw_segments_storage_base_dir, vcam_frame_rel) if pd.notna(vcam_frame_rel) else "",
            cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE, 0.25,
            vcam_yolo_detector_model.names if hasattr(vcam_yolo_detector_model, 'names') else [],
            fusion_le # target_fusion_label_encoder
        )
        preds_vcam_yolo_encoded_list.append(vcam_pred_enc)

        audio_seg_rel = row.get('audio_segment_original_file')
        audio_pred_enc, _ = get_audio_prediction_for_segment(
            audio_model,
            os.path.join(raw_segments_storage_base_dir, audio_seg_rel) if pd.notna(audio_seg_rel) else "",
            audio_scaler_yt, audio_max_len_yt,
            audio_le_yt, # audio_model_label_encoder
            fusion_le  # target_fusion_label_encoder
        )
        preds_audio_lstm_encoded_list.append(audio_pred_enc)

        fusion_pred_enc_current = -1 # Giá trị mặc định nếu lỗi
        # Kiểm tra xem index có nằm trong phạm vi của X_audio_for_fusion không
        # Điều này quan trọng nếu gt_metadata_df_test không hoàn toàn khớp 1-1 với các file features đã lưu
        # TỐT NHẤT là y_true_fusion_space_test và các X_features phải có cùng số lượng và thứ tự
        current_original_index_in_all_data = row.name # Giả sử index của gt_metadata_df_test khớp với thứ tự của features
                                                    # Hoặc bạn cần một cột ID để map

        # Lấy features tương ứng với segment hiện tại từ metadata
        # Điều này giả định rằng gt_metadata_df_test có cùng thứ tự với các file features đã lưu
        # và y_true_fusion_space_test
        if index < len(X_audio_for_fusion) and index < len(X_vcam_for_fusion):
            audio_f = X_audio_for_fusion[index:index+1]
            vcam_f = X_vcam_for_fusion[index:index+1]
            try:
                fus_probs = fusion_model.predict([audio_f, vcam_f], verbose=0)[0]
                fusion_pred_enc_current = np.argmax(fus_probs)
            except Exception as e_fus: print(f"Lỗi Fusion predict seg {row.get('segment_id', index)}: {e_fus}")
        else:
            # print(f"Cảnh báo: Index {index} nằm ngoài phạm vi của X_audio_for_fusion ({len(X_audio_for_fusion)}) hoặc X_vcam_for_fusion ({len(X_vcam_for_fusion)}). Gán BACKGROUND cho Fusion.")
            try: fusion_pred_enc_current = fusion_le.transform(["BACKGROUND"])[0]
            except: fusion_pred_enc_current = -1


        preds_fusion_encoded_list.append(fusion_pred_enc_current)
        actual_gt_labels_for_eval_list.append(gt_encoded_current_fusion_space) # Lưu GT đã mã hóa theo fusion_le

    y_true_eval_final = np.array(actual_gt_labels_for_eval_list) # Đổi tên biến
    all_predictions_final = { # Đổi tên biến
        f"VCam_YOLO_Detector_{dataset_name_suffix}": np.array(preds_vcam_yolo_encoded_list),
        f"Audio_LSTM_YT_FT_{dataset_name_suffix}": np.array(preds_audio_lstm_encoded_list),
        f"Early_Fusion_{dataset_name_suffix}": np.array(preds_fusion_encoded_list)
    }

    all_cm_paths_dict_final = {} # Đổi tên biến
    for model_name_report, y_pred_report in all_predictions_final.items():
        if len(y_pred_report) != len(y_true_eval_final):
            print(f"CẢNH BÁO: Độ dài dự đoán và GT không khớp cho {model_name_report}. Bỏ qua.")
            continue
        # Sử dụng tham số truyền vào cho đường dẫn report
        report_p = os.path.join(base_report_dir_metrics_arg, f"report_{model_name_report.lower()}.txt")
        cm_p = os.path.join(base_report_dir_figures_arg, f"cm_{model_name_report.lower()}.png")

        evaluate_and_report_single_model(y_true_eval_final, y_pred_report, model_name_report,
                                         list(fusion_le.classes_), cm_p, report_p) # SỬA Ở ĐÂY
        all_cm_paths_dict_final[model_name_report.replace(f"_{dataset_name_suffix}", "")] = cm_p


    if len(all_cm_paths_dict_final) > 1:
        combined_cm_p = os.path.join(base_report_dir_figures_arg, f"cm_comparison_all_models_{dataset_name_suffix}.png")
        # fig, axes = plt.subplots(1, len(all_cm_paths_dict_final), figsize=(8 * len(all_cm_paths_dict_final), 7))
        # if len(all_cm_paths_dict_final) == 1: axes = [axes] # Ensure axes is iterable
        # fig.suptitle(f'So sánh Ma trận Nhầm lẫn trên Tập Test {dataset_name_suffix}', fontsize=16)
        # idx_plot = 0
        # for model_title_plot, cm_file_path_plot in all_cm_paths_dict_final.items():
        #     if os.path.exists(cm_file_path_plot):
        #         img_plot = cv2.imread(cm_file_path_plot)
        #         if img_plot is not None:
        #             axes[idx_plot].imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
        #             axes[idx_plot].set_title(model_title_plot)
        #             axes[idx_plot].axis('off')
        #             idx_plot += 1
        # for i_empty in range(idx_plot, len(all_cm_paths_dict_final)): fig.delaxes(axes[i_empty])
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig(combined_cm_p); plt.close(fig)
        # print(f"Ảnh gộp CM (matplotlib) đã lưu: {combined_cm_p}")
        combine_cms_horizontally(all_cm_paths_dict_final, combined_cm_p) # Dùng hàm cũ nếu nó hoạt động


    print(f"\n===== ĐÁNH GIÁ TRÊN TẬP Test {dataset_name_suffix} HOÀN TẤT =====")

if __name__ == '__main__':
    if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories): cfg.ensure_output_directories()

    print("\n\n<<<<< ĐÁNH GIÁ TRÊN TẬP TEST IN-DOMAIN >>>>>")
    if os.path.exists(cfg.FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH) and \
       os.path.exists(cfg.FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH) and \
       os.path.exists(cfg.FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH) and \
       os.path.exists(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH):

        run_evaluation_on_dataset_files(
            dataset_name_suffix="In_Domain",
            X_audio_test_features_for_fusion_path=cfg.FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH,
            X_vcam_test_features_for_fusion_path=cfg.FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH,
            y_test_true_labels_encoded_fusion_path=cfg.FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH,
            test_segments_metadata_path=cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH,
            raw_segments_storage_base_dir=cfg.FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR, # Thư mục chứa file gốc của In-Domain Test
            audio_model_path=cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH,
            vcam_yolo_detector_path=cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH,
            fusion_model_path=cfg.BEST_FUSION_MODEL_SAVE_PATH,
            audio_youtube_scaler_path=cfg.AUDIO_YOUTUBE_SCALER_PATH,
            audio_youtube_max_len_path=cfg.AUDIO_YOUTUBE_MAX_LEN_PATH,
            audio_youtube_label_encoder_path=cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH,
            fusion_label_encoder_path=cfg.FUSION_LABEL_ENCODER_PATH,
            base_report_dir_figures_arg=cfg.FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, # Sử dụng _arg
            base_report_dir_metrics_arg=cfg.FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN  # Sử dụng _arg
        )
    else:
        print("Không tìm thấy đủ file dữ liệu Test In-Domain đã lưu hoặc file metadata In-Domain. Bỏ qua.")
        # ... (in ra các file thiếu như trước) ...

    # --- Đánh giá trên Tập Test Out-of-Domain ---
    # (Phần này giữ nguyên logic gọi run_evaluation_on_dataset_files với các đường dẫn OOD
    #  Bạn cần đảm bảo đã tạo features cho OOD tương tự như In-Domain,
    #  hoặc sửa đổi hàm run_evaluation_on_dataset_files để nó tự xử lý từ raw OOD segments nếu chưa có features)
    print("\n\n<<<<< ĐÁNH GIÁ TRÊN TẬP TEST OUT-OF-DOMAIN >>>>>")
    # Giả sử bạn đã có script `prepare_ood_evaluation_data.py` tạo ra metadata và các segment gốc
    # Và bạn cũng có một script tương tự `prepare_fusion_finetune_data.py` để tạo features cho OOD
    OOD_METADATA = getattr(cfg, 'EVALUATION_OOD_METADATA_FILE', None)
    OOD_RAW_SEGMENTS_BASE = getattr(cfg, 'EVALUATION_OOD_SEGMENTS_BASE_DIR', None)
    # Các file features .pkl cho OOD (cần được tạo bởi một pipeline riêng cho OOD features)
    OOD_AUDIO_FEATURES_PKL = getattr(cfg, 'FUSION_TEST_OOD_AUDIO_FEATURES_PATH', None) # Cần định nghĩa trong config
    OOD_VCAM_FEATURES_PKL = getattr(cfg, 'FUSION_TEST_OOD_VCAM_FEATURES_PATH', None)   # Cần định nghĩa trong config
    OOD_TRUE_LABELS_PKL = getattr(cfg, 'FUSION_TEST_OOD_TRUE_LABELS_PATH', None)     # Cần định nghĩa trong config

    if OOD_METADATA and os.path.exists(OOD_METADATA) and \
       OOD_RAW_SEGMENTS_BASE and \
       OOD_AUDIO_FEATURES_PKL and os.path.exists(OOD_AUDIO_FEATURES_PKL) and \
       OOD_VCAM_FEATURES_PKL and os.path.exists(OOD_VCAM_FEATURES_PKL) and \
       OOD_TRUE_LABELS_PKL and os.path.exists(OOD_TRUE_LABELS_PKL):

        run_evaluation_on_dataset_files(
            dataset_name_suffix="OOD",
            X_audio_test_features_for_fusion_path=OOD_AUDIO_FEATURES_PKL,
            X_vcam_test_features_for_fusion_path=OOD_VCAM_FEATURES_PKL,
            y_test_true_labels_encoded_fusion_path=OOD_TRUE_LABELS_PKL,
            test_segments_metadata_path=OOD_METADATA,
            raw_segments_storage_base_dir=OOD_RAW_SEGMENTS_BASE,
            audio_model_path=cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH,
            vcam_yolo_detector_path=cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH,
            fusion_model_path=cfg.BEST_FUSION_MODEL_SAVE_PATH,
            audio_youtube_scaler_path=cfg.AUDIO_YOUTUBE_SCALER_PATH,
            audio_youtube_max_len_path=cfg.AUDIO_YOUTUBE_MAX_LEN_PATH,
            audio_youtube_label_encoder_path=cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH,
            fusion_label_encoder_path=cfg.FUSION_LABEL_ENCODER_PATH,
            base_report_dir_figures_arg=cfg.EVALUATION_OOD_REPORTS_FIGURES_DIR, # Sử dụng _arg
            base_report_dir_metrics_arg=cfg.EVALUATION_OOD_REPORTS_METRICS_DIR  # Sử dụng _arg
        )
    else:
        print("Không tìm thấy đủ file dữ liệu Test Out-Of-Domain đã xử lý/cấu hình (features, labels, metadata). Bỏ qua.")
        print("Vui lòng chạy script chuẩn bị dữ liệu và features cho tập OOD và cập nhật config.")
        if not (OOD_AUDIO_FEATURES_PKL and os.path.exists(OOD_AUDIO_FEATURES_PKL)): print(f"  Thiếu OOD Audio Features: {OOD_AUDIO_FEATURES_PKL}")
        if not (OOD_VCAM_FEATURES_PKL and os.path.exists(OOD_VCAM_FEATURES_PKL)): print(f"  Thiếu OOD VCam Features: {OOD_VCAM_FEATURES_PKL}")
        if not (OOD_TRUE_LABELS_PKL and os.path.exists(OOD_TRUE_LABELS_PKL)): print(f"  Thiếu OOD True Labels: {OOD_TRUE_LABELS_PKL}")
        if not (OOD_METADATA and os.path.exists(OOD_METADATA)): print(f"  Thiếu OOD Metadata: {OOD_METADATA}")


    print("\n===== ĐÁNH GIÁ TẤT CẢ CÁC MÔ HÌNH HOÀN TẤT (Hoặc bỏ qua nếu thiếu dữ liệu) =====")