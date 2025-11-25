# Drone_Detection_Project/src/evaluation/evaluate_all_models_on_ood.py
import os
import pickle
import numpy as np
import librosa
import cv2 # Cho VCam
import torch # Cho VCam YOLO và Feature Extractor
import torch.nn as nn # Cho FeatureExtractorVCam
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import từ các module trong src
try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    # Giả sử FeatureExtractorVCam và preprocess_frame_for_yolo_input nằm trong module này
    from src.data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
except ImportError:
    import sys
    current_dir_eval_all = os.path.dirname(os.path.abspath(__file__))
    src_dir_eval_all = os.path.dirname(current_dir_eval_all)
    project_root_eval_all = os.path.dirname(src_dir_eval_all)
    if project_root_eval_all not in sys.path: sys.path.insert(0, project_root_eval_all)
    if src_dir_eval_all not in sys.path: sys.path.insert(0, src_dir_eval_all)
    from config_loader.loader import get_config
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input
    from evaluation.plotting_utils import plot_custom_confusion_matrix

cfg_eval_all_ood = get_config()
device_torch_eval_ood = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_vcam_yolo_classification_for_ood_segment(
    yolo_model, image_path, target_imgsz,
    yolo_detection_conf_thresh,
    yolo_class_names_from_model, # Tên lớp từ model YOLO (ví dụ 4 lớp object)
    master_class_list_for_report # Danh sách 5 lớp bạn muốn báo cáo (bao gồm BACKGROUND)
):
    """
    Chạy YOLO trên ảnh, nếu có phát hiện đối tượng thì trả về lớp đó.
    Nếu không có phát hiện nào đạt ngưỡng, trả về "BACKGROUND".
    """
    if not os.path.exists(image_path):
        # print(f"Cảnh báo: Ảnh không tồn tại {image_path} cho VCam predict.")
        return "BACKGROUND" # Mặc định nếu ảnh không có
    try:
        frame_cv = cv2.imread(image_path)
        if frame_cv is None:
            # print(f"Cảnh báo: Không thể đọc ảnh {image_path} cho VCam predict.")
            return "BACKGROUND"
    except Exception as e_read_img:
        # print(f"Lỗi đọc ảnh {image_path}: {e_read_img}")
        return "BACKGROUND"

    predictions = yolo_model.predict(source=frame_cv, imgsz=target_imgsz, conf=yolo_detection_conf_thresh, verbose=False, device=device_torch_eval_ood)
    
    best_detected_object_class_name = None
    max_conf = 0.0

    if predictions and predictions[0].boxes and hasattr(predictions[0].boxes, 'conf') and len(predictions[0].boxes.conf) > 0:
        for box_idx in range(len(predictions[0].boxes.conf)):
            conf = float(predictions[0].boxes.conf[box_idx])
            cls_id = int(predictions[0].boxes.cls[box_idx])
            
            if 0 <= cls_id < len(yolo_class_names_from_model):
                detected_class_name_by_yolo = yolo_class_names_from_model[cls_id]
                # Chỉ xem xét các lớp đối tượng thực sự, không phải lớp "BACKGROUND" mà YOLO có thể học (nếu có)
                # và lớp đó phải có trong master list
                if detected_class_name_by_yolo in master_class_list_for_report and detected_class_name_by_yolo != "BACKGROUND":
                    if conf > max_conf:
                        max_conf = conf
                        best_detected_object_class_name = detected_class_name_by_yolo
            # else:
            #     print(f"Cảnh báo: cls_id {cls_id} từ YOLO cho {os.path.basename(image_path)} nằm ngoài phạm vi yolo_class_names_from_model ({len(yolo_class_names_from_model)} lớp).")

    if best_detected_object_class_name is not None:
        return best_detected_object_class_name
    else:
        return "BACKGROUND" # Nếu không có phát hiện đối tượng nào đạt ngưỡng


def run_all_models_evaluation_on_ood():
    print("===== STARTING EVALUATION OF ALL MODELS ON OUT-OF-DOMAIN (OOD) TEST SET =====")
    if hasattr(cfg_eval_all_ood, 'ensure_output_directories') and callable(cfg_eval_all_ood.ensure_output_directories):
        cfg_eval_all_ood.ensure_output_directories()
    else: # Tự tạo các thư mục báo cáo OOD
        os.makedirs(cfg_eval_all_ood.EVALUATION_OOD_REPORTS_METRICS_DIR, exist_ok=True)
        os.makedirs(cfg_eval_all_ood.EVALUATION_OOD_REPORTS_FIGURES_DIR, exist_ok=True)


    # 1. Kiểm tra và tải Metadata OOD
    print(f"Đang tải OOD Ground Truth Metadata từ: {cfg_eval_all_ood.EVALUATION_OOD_METADATA_FILE}")
    if not os.path.exists(cfg_eval_all_ood.EVALUATION_OOD_METADATA_FILE):
        print(f"LỖI: File metadata OOD không tồn tại. Vui lòng chạy 'prepare_ood_evaluation_data.py' trước.")
        return
    try:
        ood_gt_metadata_df = pd.read_csv(cfg_eval_all_ood.EVALUATION_OOD_METADATA_FILE)
        print(f"Đã tải {len(ood_gt_metadata_df)} segment OOD ground truth.")
    except Exception as e:
        print(f"Lỗi khi đọc file metadata OOD: {e}"); return

    # LabelEncoder cho 5 lớp (MASTER_CLASS_LIST_FUSION)
    master_label_encoder = LabelEncoder()
    master_label_encoder.fit(cfg_eval_all_ood.MASTER_CLASS_LIST_FUSION)
    class_names_for_master_report = list(master_label_encoder.classes_)
    
    ground_truth_labels_text = []
    valid_indices_for_eval = [] # Lưu index của các dòng metadata hợp lệ
    for idx, gt_label in enumerate(ood_gt_metadata_df['ground_truth_label']):
        try:
            # Kiểm tra xem nhãn GT có trong master list không
            if str(gt_label).strip() in class_names_for_master_report:
                ground_truth_labels_text.append(str(gt_label).strip())
                valid_indices_for_eval.append(idx)
            # else:
                # print(f"Cảnh báo: Nhãn GT '{gt_label}' trong OOD metadata không thuộc MASTER_CLASS_LIST_FUSION. Bỏ qua segment này.")
        except Exception as e_label:
            # print(f"Cảnh báo: Lỗi xử lý nhãn GT '{gt_label}': {e_label}. Bỏ qua.")
            pass
    
    if not ground_truth_labels_text:
        print("Lỗi: Không có nhãn ground truth hợp lệ nào trong OOD metadata. Dừng.")
        return
        
    ood_gt_metadata_df_filtered = ood_gt_metadata_df.iloc[valid_indices_for_eval].copy() # Chỉ giữ lại các dòng hợp lệ
    ground_truth_encoded_master = master_label_encoder.transform(ground_truth_labels_text)
    print(f"Số segment OOD hợp lệ để đánh giá: {len(ground_truth_encoded_master)}")

    # 2. Tải các Mô hình và Thành phần Tiền xử lý
    print("\n--- Đang tải các mô hình và thành phần tiền xử lý ---")
    try:
        # VCam YOLO Fine-tuned
        vcam_yolo_ft_model = YOLO(cfg_eval_all_ood.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        # vcam_yolo_ft_model.to(device_torch_eval_ood) # Sẽ truyền device vào predict
        yolo_model_class_names_actual = [] # Các lớp mà YOLO này thực sự được train để detect
        if hasattr(vcam_yolo_ft_model, 'names'):
            if isinstance(vcam_yolo_ft_model.names, dict): yolo_model_class_names_actual = [vcam_yolo_ft_model.names[i] for i in sorted(vcam_yolo_ft_model.names.keys())]
            elif isinstance(vcam_yolo_ft_model.names, list): yolo_model_class_names_actual = vcam_yolo_ft_model.names
        print(f"  Đã tải VCam YOLO Fine-tuned. Các lớp model có thể detect: {yolo_model_class_names_actual}")

        # Audio LSTM Fine-tuned
        audio_lstm_ft_model = load_model(cfg_eval_all_ood.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
        with open(cfg_eval_all_ood.AUDIO_YOUTUBE_SCALER_PATH, 'rb') as f: audio_ft_scaler = pickle.load(f)
        with open(cfg_eval_all_ood.AUDIO_YOUTUBE_MAX_LEN_PATH, 'rb') as f: audio_ft_max_len = pickle.load(f)
        # audio_ft_label_encoder không cần thiết ở đây vì ta dùng master_label_encoder
        print(f"  Đã tải Audio LSTM Fine-tuned, Scaler (max_len={audio_ft_max_len}).")

        # Early Fusion Model
        fusion_model = load_model(cfg_eval_all_ood.BEST_FUSION_MODEL_SAVE_PATH)
        # Cần VCam Feature Extractor cho Fusion
        vcam_yolo_for_feat_ext_fusion = YOLO(cfg_eval_all_ood.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION) # Model VCam YT FT
        if hasattr(vcam_yolo_for_feat_ext_fusion, 'model') and hasattr(vcam_yolo_for_feat_ext_fusion.model, 'model'):
            yolo_sequential_part_fusion = vcam_yolo_for_feat_ext_fusion.model.model
        else: raise ValueError("Không thể truy cập backbone của VCam model cho feature extraction (fusion).")
        vcam_feat_extractor_for_fusion = FeatureExtractorVCam(yolo_sequential_part_fusion, extraction_layer_index=8) # Giả sử tầng 8
        vcam_feat_extractor_for_fusion.to(device_torch_eval_ood).eval()
        print(f"  Đã tải Early Fusion Model và VCam Feature Extractor (cho fusion) trên {device_torch_eval_ood}.")

    except Exception as e:
        print(f"LỖI khi tải một trong các mô hình hoặc thành phần: {e}"); import traceback; traceback.print_exc(); return

    # 3. Thực hiện dự đoán trên tập OOD
    vcam_yolo_preds_text = []
    audio_lstm_preds_encoded = [] # Sẽ chuyển sang text sau
    fusion_model_preds_encoded = [] # Sẽ chuyển sang text sau

    print("\n--- Thực hiện dự đoán trên tập OOD ---")
    for index, row in tqdm(ood_gt_metadata_df_filtered.iterrows(), total=ood_gt_metadata_df_filtered.shape[0], desc="Predicting on OOD Set"):
        # Đường dẫn tương đối từ ood_processed_segments_base_dir
        vcam_frame_rel_path = row.get('path_to_vcam_frame')
        audio_segment_rel_path = row.get('path_to_audio_segment')

        # a. Dự đoán bằng VCam YOLO Fine-tuned (cho ra 1 trong 5 lớp master)
        vcam_pred_class_name = "BACKGROUND" # Mặc định
        if pd.notna(vcam_frame_rel_path):
            vcam_frame_abs_path = os.path.join(cfg_eval_all_ood.EVALUATION_OOD_SEGMENTS_BASE_DIR, vcam_frame_rel_path)
            vcam_pred_class_name = get_vcam_yolo_classification_for_ood_segment(
                vcam_yolo_ft_model, vcam_frame_abs_path, cfg_eval_all_ood.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                0.25, yolo_model_class_names_actual, class_names_for_master_report
            )
        vcam_yolo_preds_text.append(vcam_pred_class_name)

        # b. Dự đoán bằng Audio LSTM Fine-tuned
        audio_pred_idx = master_label_encoder.transform(["BACKGROUND"])[0] # Mặc định là BACKGROUND
        if pd.notna(audio_segment_rel_path):
            audio_segment_abs_path = os.path.join(cfg_eval_all_ood.EVALUATION_OOD_SEGMENTS_BASE_DIR, audio_segment_rel_path)
            if os.path.exists(audio_segment_abs_path):
                try:
                    audio_data, _ = librosa.load(audio_segment_abs_path, sr=cfg_eval_all_ood.AUDIO_SAMPLE_RATE)
                    if len(audio_data) >= int(cfg_eval_all_ood.AUDIO_SEGMENT_DURATION * cfg_eval_all_ood.AUDIO_SAMPLE_RATE * 0.1):
                        mfcc_s = extract_mfcc(audio_data, cfg_eval_all_ood.AUDIO_SAMPLE_RATE, cfg_eval_all_ood.AUDIO_N_MFCC, cfg_eval_all_ood.AUDIO_N_FFT, cfg_eval_all_ood.AUDIO_HOP_LENGTH)
                        mfcc_p, _ = pad_features([mfcc_s], max_len=audio_ft_max_len)
                        if mfcc_p.size > 0:
                            mfcc_sc, _ = scale_features(mfcc_p, scaler=audio_ft_scaler)
                            if mfcc_sc.size > 0:
                                audio_probs = audio_lstm_ft_model.predict(mfcc_sc, verbose=0)[0]
                                audio_pred_idx = np.argmax(audio_probs) # Đây là index dựa trên output của audio_lstm_ft_model
                                # Cần map index này về master_label_encoder nếu số lớp khác nhau
                                # Giả sử audio_lstm_ft_model cũng output 5 lớp theo thứ tự của MASTER_CLASS_LIST_FUSION
                except Exception as e_audio_pred:
                    print(f"Lỗi predict audio cho {audio_segment_rel_path}: {e_audio_pred}")
        audio_lstm_preds_encoded.append(audio_pred_idx)


        # c. Dự đoán bằng Fusion Model
        fusion_pred_idx = master_label_encoder.transform(["BACKGROUND"])[0] # Mặc định
        vcam_feat_for_fusion_np = np.zeros((1, cfg_eval_all_ood.FUSION_VCAM_FEATURE_DIM), dtype=np.float32)
        audio_feat_for_fusion_np = np.zeros((1, audio_ft_max_len, cfg_eval_all_ood.AUDIO_N_MFCC), dtype=np.float32) # Tạo MFCC rỗng

        # Trích xuất VCam feature cho fusion
        if pd.notna(vcam_frame_rel_path):
            vcam_frame_abs_path_fusion = os.path.join(cfg_eval_all_ood.EVALUATION_OOD_SEGMENTS_BASE_DIR, vcam_frame_rel_path)
            if os.path.exists(vcam_frame_abs_path_fusion):
                frame_cv_fusion = cv2.imread(vcam_frame_abs_path_fusion)
                if frame_cv_fusion is not None:
                    vcam_tensor = preprocess_frame_for_yolo_input(frame_cv_fusion, cfg_eval_all_ood.VCAM_YOLO_IMG_SIZE) # Dùng imgsz của model tạo feature extractor
                    if vcam_tensor is not None:
                        with torch.no_grad():
                            vcam_feat_for_fusion_np = vcam_feat_extractor_for_fusion(vcam_tensor.to(device_torch_eval_ood)).cpu().numpy()
        
        # Lấy Audio feature cho fusion (MFCC đã xử lý)
        if pd.notna(audio_segment_rel_path) and 'mfcc_sc' in locals() and mfcc_sc.size > 0 : # Nếu audio được xử lý ở bước b
             audio_feat_for_fusion_np = mfcc_sc # mfcc_sc đã có shape (1, max_len, n_mfcc)
        # Nếu không, audio_feat_for_fusion_np vẫn là zero_array (xử lý trường hợp audio bị lỗi/thiếu)

        try:
            fusion_probs = fusion_model.predict([audio_feat_for_fusion_np, vcam_feat_for_fusion_np], verbose=0)[0]
            fusion_pred_idx = np.argmax(fusion_probs)
        except Exception as e_fusion_pred:
            print(f"Lỗi predict fusion cho segment {row.get('segment_id', index)}: {e_fusion_pred}")
        fusion_model_preds_encoded.append(fusion_pred_idx)


    # 4. Chuyển đổi tất cả dự đoán về dạng số nguyên theo master_label_encoder
    vcam_yolo_preds_encoded_master = master_label_encoder.transform(vcam_yolo_preds_text)
    # audio_lstm_preds_encoded và fusion_model_preds_encoded đã là index theo output của model
    # GIẢ SỬ output của audio_ft_model và fusion_model có thứ tự lớp giống MASTER_CLASS_LIST_FUSION

    # 5. Tạo và Lưu Báo cáo So sánh
    print("\n--- KẾT QUẢ ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH TRÊN TẬP OOD ---")
    models_to_evaluate_ood = {
        "VCam_YOLO_FT_OOD": np.array(vcam_yolo_preds_encoded_master),
        "Audio_LSTM_FT_OOD": np.array(audio_lstm_preds_encoded), # Giả sử đã khớp với master encoder
        "Early_Fusion_OOD": np.array(fusion_model_preds_encoded) # Giả sử đã khớp
    }

    full_classification_report_str_ood = ""
    all_reports_data = {} # Để lưu dữ liệu report nếu cần phân tích thêm

    for model_name, preds_encoded in models_to_evaluate_ood.items():
        print(f"\n--- Kết quả cho Mô hình: {model_name} trên OOD ---")
        if len(preds_encoded) != len(ground_truth_encoded_master):
            print(f"  CẢNH BÁO: Số lượng dự đoán ({len(preds_encoded)}) không khớp với ground truth ({len(ground_truth_encoded_master)}) cho {model_name}.")
            continue
        
        report = classification_report(ground_truth_encoded_master, preds_encoded,
                                       target_names=class_names_for_master_report,
                                       labels=range(len(class_names_for_master_report)), # labels là ID từ 0 đến N-1
                                       output_dict=True, # Lấy report dạng dict để dễ xử lý
                                       zero_division=0)
        print(classification_report(ground_truth_encoded_master, preds_encoded,
                                   target_names=class_names_for_master_report,
                                   labels=range(len(class_names_for_master_report)),
                                   zero_division=0)) # In ra console
        
        all_reports_data[model_name] = report # Lưu report dạng dict
        
        # Tạo chuỗi report để lưu file text
        # Convert report dict to string manually for better formatting if needed
        report_string_for_file = classification_report(
            ground_truth_encoded_master, preds_encoded,
            target_names=class_names_for_master_report,
            labels=range(len(class_names_for_master_report)),
            zero_division=0
        )
        full_classification_report_str_ood += f"===== {model_name} =====\n{report_string_for_file}\n\n"

        # Vẽ và lưu CM
        cm_filename_ood = os.path.join(cfg_eval_all_ood.EVALUATION_OOD_REPORTS_FIGURES_DIR, f"cm_{model_name.lower()}.png")
        plot_custom_confusion_matrix(ground_truth_encoded_master, preds_encoded, class_names_for_master_report,
                                     filename=cm_filename_ood, title=f"CM - {model_name}")

    # Lưu report tổng hợp
    report_file_path_ood = os.path.join(cfg_eval_all_ood.EVALUATION_OOD_REPORTS_METRICS_DIR, "all_models_classification_reports_ood.txt")
    with open(report_file_path_ood, 'w') as f: f.write(full_classification_report_str_ood)
    print(f"\nBáo cáo phân loại tổng hợp cho OOD đã lưu vào: {report_file_path_ood}")

    # (Tùy chọn) Vẽ biểu đồ so sánh các chỉ số chính (ví dụ F1-score weighted)
    # Bạn có thể trích xuất từ all_reports_data và vẽ bằng matplotlib/seaborn

    print("===== EVALUATION OF ALL MODELS ON OOD TEST SET FINISHED =====")


if __name__ == '__main__':
    # Kiểm tra các file đầu vào chính
    required_main_files = [
        cfg_eval_all_ood.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE,
        cfg_eval_all_ood.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH,
        cfg_eval_all_ood.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH,
        cfg_eval_all_ood.BEST_FUSION_MODEL_SAVE_PATH
    ]
    missing_main = [f for f in required_main_files if not os.path.exists(f)]
    if missing_main:
        print("LỖI: Thiếu các file mô hình hoặc metadata OOD cần thiết:")
        for f_path in missing_main: print(f" - {f_path}")
    else:
        run_all_models_evaluation_on_ood()
        