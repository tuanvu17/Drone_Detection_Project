# Drone_Detection_Project/src/analysis/generate_comprehensive_evaluation_report.py
"""
Script tổng hợp để tạo báo cáo đánh giá đầy đủ theo yêu cầu:
- TABLE I: Overall Comparison on OOD
- TABLE II: Per-Class Performance  
- Confusion Matrices
- Training Settings

Sử dụng lại code từ evaluate_all_models_on_ood.py và tổng hợp kết quả
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import librosa
import yaml

# Thêm đường dẫn project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import project_config as cfg
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
except ImportError as e:
    print(f"Lỗi import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU mode
device_torch = torch.device("cpu")


def get_vcam_yolo_classification_for_ood_segment(
    yolo_model, image_path, target_imgsz,
    yolo_detection_conf_thresh,
    yolo_class_names_from_model,
    master_class_list_for_report
):
    """Lấy dự đoán từ VCam YOLO (copy từ evaluate_all_models_on_ood.py)"""
    if not os.path.exists(image_path):
        return "BACKGROUND"
    try:
        frame_cv = cv2.imread(image_path)
        if frame_cv is None:
            return "BACKGROUND"
    except Exception:
        return "BACKGROUND"

    predictions = yolo_model.predict(source=frame_cv, imgsz=target_imgsz, 
                                    conf=yolo_detection_conf_thresh, verbose=False, 
                                    device=device_torch)
    
    best_detected_object_class_name = None
    max_conf = 0.0

    if predictions and predictions[0].boxes and hasattr(predictions[0].boxes, 'conf') and len(predictions[0].boxes.conf) > 0:
        for box_idx in range(len(predictions[0].boxes.conf)):
            conf = float(predictions[0].boxes.conf[box_idx])
            cls_id = int(predictions[0].boxes.cls[box_idx])
            
            if 0 <= cls_id < len(yolo_class_names_from_model):
                detected_class_name_by_yolo = yolo_class_names_from_model[cls_id]
                if detected_class_name_by_yolo in master_class_list_for_report and detected_class_name_by_yolo != "BACKGROUND":
                    if conf > max_conf:
                        max_conf = conf
                        best_detected_object_class_name = detected_class_name_by_yolo

    if best_detected_object_class_name is not None:
        return best_detected_object_class_name
    else:
        return "BACKGROUND"


def evaluate_all_models_on_ood_for_report():
    """
    Đánh giá tất cả models trên OOD và trả về kết quả chi tiết
    (Dựa trên evaluate_all_models_on_ood.py)
    """
    print("="*80)
    print("ĐÁNH GIÁ TẤT CẢ MODELS TRÊN OOD TEST SET")
    print("="*80)
    
    # Kiểm tra OOD metadata - sử dụng EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE
    ood_metadata_path = getattr(cfg, 'EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE', None)
    if not ood_metadata_path or not os.path.exists(ood_metadata_path):
        # Thử đường dẫn khác
        ood_metadata_path = os.path.join(cfg.PROJECT_ROOT, 'data', 'processed', 
                                        'Evaluation_Test_Segments', 'evaluation_test_ground_truth.csv')
        if not os.path.exists(ood_metadata_path):
            print(f"LỖI: File OOD metadata không tồn tại")
            print(f"  Đã thử: {getattr(cfg, 'EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE', 'N/A')}")
            print(f"  Đã thử: {ood_metadata_path}")
            return None
    
    # Load OOD metadata
    print(f"\nĐang tải OOD metadata từ: {ood_metadata_path}")
    ood_df = pd.read_csv(ood_metadata_path)
    print(f"  Tổng số segments: {len(ood_df)}")
    
    # Master label encoder
    master_label_encoder = LabelEncoder()
    master_label_encoder.fit(cfg.MASTER_CLASS_LIST_FUSION)
    class_names = list(master_label_encoder.classes_)
    print(f"  Classes: {class_names}")
    
    # Lọc các nhãn hợp lệ
    ground_truth_labels_text = []
    valid_indices = []
    for idx, gt_label in enumerate(ood_df['ground_truth_label']):
        try:
            if str(gt_label).strip() in class_names:
                ground_truth_labels_text.append(str(gt_label).strip())
                valid_indices.append(idx)
        except:
            pass
    
    if not ground_truth_labels_text:
        print("Lỗi: Không có nhãn ground truth hợp lệ")
        return None
    
    ood_df_filtered = ood_df.iloc[valid_indices].copy()
    ground_truth_encoded = master_label_encoder.transform(ground_truth_labels_text)
    print(f"  Số segment hợp lệ: {len(ground_truth_encoded)}")
    
    # Load models
    print("\nĐang tải các models...")
    
    # VCam YOLO
    vcam_model_path = cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH
    if not os.path.exists(vcam_model_path):
        print(f"LỖI: VCam model không tồn tại: {vcam_model_path}")
        return None
    vcam_yolo = YOLO(vcam_model_path)
    yolo_class_names = []
    if hasattr(vcam_yolo, 'names'):
        if isinstance(vcam_yolo.names, dict):
            yolo_class_names = [vcam_yolo.names[i] for i in sorted(vcam_yolo.names.keys())]
        elif isinstance(vcam_yolo.names, list):
            yolo_class_names = vcam_yolo.names
    print(f"  ✓ VCam YOLO loaded (classes: {yolo_class_names})")
    
    # Audio model
    audio_model_path = cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH
    audio_scaler_path = cfg.AUDIO_YOUTUBE_SCALER_PATH
    audio_max_len_path = cfg.AUDIO_YOUTUBE_MAX_LEN_PATH
    audio_label_encoder_path = cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH
    
    if not all(os.path.exists(p) for p in [audio_model_path, audio_scaler_path, 
                                           audio_max_len_path, audio_label_encoder_path]):
        print(f"LỖI: Thiếu file Audio model hoặc components")
        return None
    
    audio_model = load_model(audio_model_path)
    with open(audio_scaler_path, 'rb') as f:
        audio_scaler = pickle.load(f)
    with open(audio_max_len_path, 'rb') as f:
        audio_max_len = pickle.load(f)
    with open(audio_label_encoder_path, 'rb') as f:
        audio_label_encoder = pickle.load(f)
    print(f"  ✓ Audio model loaded")
    
    # Fusion model
    fusion_model_path = cfg.BEST_FUSION_MODEL_SAVE_PATH
    fusion_label_encoder_path = cfg.FUSION_LABEL_ENCODER_PATH
    
    if not all(os.path.exists(p) for p in [fusion_model_path, fusion_label_encoder_path]):
        print(f"LỖI: Thiếu file Fusion model hoặc components")
        return None
    
    fusion_model = load_model(fusion_model_path)
    with open(fusion_label_encoder_path, 'rb') as f:
        fusion_label_encoder = pickle.load(f)
    
    # VCam feature extractor for fusion
    try:
        vcam_yolo_for_feat = YOLO(cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION)
        if hasattr(vcam_yolo_for_feat, 'model') and hasattr(vcam_yolo_for_feat.model, 'model'):
            yolo_sequential_part = vcam_yolo_for_feat.model.model
        else:
            raise ValueError("Không thể truy cập backbone của VCam model")
        vcam_feat_extractor = FeatureExtractorVCam(yolo_sequential_part, extraction_layer_index=8)
        vcam_feat_extractor.to(device_torch).eval()
        print(f"  ✓ Fusion model loaded")
    except Exception as e:
        print(f"  Cảnh báo: Không thể tạo VCam feature extractor: {e}")
        vcam_feat_extractor = None
    
    # OOD segments base directory
    ood_segments_dir = getattr(cfg, 'EVALUATION_TEST_SEGMENTS_BASE_DIR', 
                               os.path.join(cfg.PROJECT_ROOT, 'data', 'processed', 'Evaluation_Test_Segments'))
    
    # Dự đoán
    print(f"\nĐang dự đoán trên {len(ood_df_filtered)} segments...")
    
    vcam_predictions_text = []
    audio_predictions_encoded = []
    fusion_predictions_encoded = []
    
    for idx, row in tqdm(ood_df_filtered.iterrows(), total=len(ood_df_filtered), desc="Predicting"):
        # VCam prediction
        vcam_frame_rel_path = row.get('path_to_vcam_frame', '')
        if pd.notna(vcam_frame_rel_path):
            vcam_frame_abs_path = os.path.join(ood_segments_dir, vcam_frame_rel_path)
            vcam_pred = get_vcam_yolo_classification_for_ood_segment(
                vcam_yolo, vcam_frame_abs_path, cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                0.25, yolo_class_names, class_names
            )
        else:
            vcam_pred = "BACKGROUND"
        vcam_predictions_text.append(vcam_pred)
        
        # Audio prediction
        audio_segment_rel_path = row.get('path_to_audio_segment', '')
        audio_pred_idx = master_label_encoder.transform(["BACKGROUND"])[0]
        mfcc_sc = None
        
        if pd.notna(audio_segment_rel_path):
            audio_segment_abs_path = os.path.join(ood_segments_dir, audio_segment_rel_path)
            if os.path.exists(audio_segment_abs_path):
                try:
                    audio_data, _ = librosa.load(audio_segment_abs_path, sr=cfg.AUDIO_SAMPLE_RATE)
                    if len(audio_data) >= int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.1):
                        mfcc_s = extract_mfcc(audio_data, cfg.AUDIO_SAMPLE_RATE, 
                                            cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, 
                                            cfg.AUDIO_HOP_LENGTH)
                        mfcc_p, _ = pad_features([mfcc_s], max_len=audio_max_len)
                        if mfcc_p.size > 0:
                            mfcc_sc, _ = scale_features(mfcc_p, scaler=audio_scaler)
                            if mfcc_sc.size > 0:
                                audio_probs = audio_model.predict(mfcc_sc, verbose=0)[0]
                                audio_pred_idx = np.argmax(audio_probs)
                except Exception as e:
                    pass
        audio_predictions_encoded.append(audio_pred_idx)
        
        # Fusion prediction
        fusion_pred_idx = master_label_encoder.transform(["BACKGROUND"])[0]
        vcam_feat_for_fusion = np.zeros((1, cfg.FUSION_VCAM_FEATURE_DIM), dtype=np.float32)
        audio_feat_for_fusion = np.zeros((1, audio_max_len, cfg.AUDIO_N_MFCC), dtype=np.float32)
        
        # VCam feature
        if pd.notna(vcam_frame_rel_path) and vcam_feat_extractor is not None:
            vcam_frame_abs_path = os.path.join(ood_segments_dir, vcam_frame_rel_path)
            if os.path.exists(vcam_frame_abs_path):
                frame_cv = cv2.imread(vcam_frame_abs_path)
                if frame_cv is not None:
                    vcam_tensor = preprocess_frame_for_yolo_input(frame_cv, cfg.VCAM_YOLO_IMG_SIZE)
                    if vcam_tensor is not None:
                        with torch.no_grad():
                            vcam_feat_for_fusion = vcam_feat_extractor(vcam_tensor.to(device_torch)).cpu().numpy()
        
        # Audio feature
        if mfcc_sc is not None and mfcc_sc.size > 0:
            audio_feat_for_fusion = mfcc_sc
        
        try:
            fusion_probs = fusion_model.predict([audio_feat_for_fusion, vcam_feat_for_fusion], verbose=0)[0]
            fusion_pred_idx = np.argmax(fusion_probs)
        except:
            pass
        fusion_predictions_encoded.append(fusion_pred_idx)
    
    # Encode predictions
    vcam_encoded = master_label_encoder.transform(vcam_predictions_text)
    
    print(f"\nĐã xử lý {len(ground_truth_encoded)} segments hợp lệ")
    
    return {
        'ground_truth': ground_truth_encoded,
        'vcam_predictions': vcam_encoded,
        'audio_predictions': np.array(audio_predictions_encoded),
        'fusion_predictions': np.array(fusion_predictions_encoded),
        'class_names': class_names,
        'label_encoder': master_label_encoder
    }


def calculate_metrics(y_true, y_pred, class_names):
    """Tính toán các metrics: Accuracy, Precision, Recall, F1-Score"""
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    # Weighted averages
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    }


def generate_table_i(results_dict, output_path):
    """Tạo TABLE I: Overall Comparison on OOD"""
    print("\n" + "="*80)
    print("TABLE I: OVERALL COMPARISON ON OOD")
    print("="*80)
    
    table_data = []
    model_mapping = {
        'vcam_predictions': 'VCam Fine-tuned',
        'audio_predictions': 'Audio Fine-tuned',
        'fusion_predictions': 'Early Fusion'
    }
    
    for pred_key, model_name in model_mapping.items():
        if pred_key not in results_dict:
            continue
        
        metrics = calculate_metrics(
            results_dict['ground_truth'],
            results_dict[pred_key],
            results_dict['class_names']
        )
        
        table_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision_weighted']:.4f}",
            'Recall': f"{metrics['recall_weighted']:.4f}",
            'F1-Score': f"{metrics['f1_weighted']:.4f}"
        })
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall: {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score: {metrics['f1_weighted']:.4f}")
    
    # Lưu CSV
    df = pd.DataFrame(table_data)
    df.to_csv(output_path, index=False)
    print(f"\nTABLE I đã lưu: {output_path}")
    
    return table_data, df


def generate_table_ii(results_dict, output_path):
    """Tạo TABLE II: Per-Class Performance"""
    print("\n" + "="*80)
    print("TABLE II: PER-CLASS PERFORMANCE")
    print("="*80)
    
    class_names = results_dict['class_names']
    model_mapping = {
        'vcam_predictions': 'VCam Fine-tuned',
        'audio_predictions': 'Audio Fine-tuned',
        'fusion_predictions': 'Early Fusion'
    }
    
    table_data = []
    
    for pred_key, model_name in model_mapping.items():
        if pred_key not in results_dict:
            continue
        
        metrics = calculate_metrics(
            results_dict['ground_truth'],
            results_dict[pred_key],
            class_names
        )
        
        per_class = metrics['per_class']
        
        for i, class_name in enumerate(class_names):
            table_data.append({
                'Model': model_name,
                'Class': class_name,
                'Precision': f"{per_class['precision'][i]:.4f}",
                'Recall': f"{per_class['recall'][i]:.4f}",
                'F1-Score': f"{per_class['f1'][i]:.4f}",
                'Support': int(per_class['support'][i])
            })
            
            print(f"\n{model_name} - {class_name}:")
            print(f"  Precision: {per_class['precision'][i]:.4f}")
            print(f"  Recall: {per_class['recall'][i]:.4f}")
            print(f"  F1-Score: {per_class['f1'][i]:.4f}")
            print(f"  Support: {int(per_class['support'][i])}")
    
    # Lưu CSV
    df = pd.DataFrame(table_data)
    df.to_csv(output_path, index=False)
    print(f"\nTABLE II đã lưu: {output_path}")
    
    return table_data, df


def generate_confusion_matrices(results_dict, output_dir):
    """Tạo Confusion Matrices cho 3 models"""
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES")
    print("="*80)
    
    class_names = results_dict['class_names']
    model_mapping = {
        'vcam_predictions': ('VCam Fine-tuned', 'vcam'),
        'audio_predictions': ('Audio Fine-tuned', 'audio'),
        'fusion_predictions': ('Early Fusion', 'fusion')
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for pred_key, (model_name, model_short) in model_mapping.items():
        if pred_key not in results_dict:
            continue
        
        cm_path = os.path.join(output_dir, f'confusion_matrix_{model_short}.png')
        
        plot_custom_confusion_matrix(
            results_dict['ground_truth'],
            results_dict[pred_key],
            class_names,
            filename=cm_path,
            title=f"Confusion Matrix - {model_name}"
        )
        
        print(f"  ✓ {model_name}: {cm_path}")


def collect_training_settings():
    """Thu thập Training Settings từ config và args.yaml files"""
    print("\n" + "="*80)
    print("TRAINING SETTINGS")
    print("="*80)
    
    settings = {}
    
    # VCam Fine-tuned settings
    vcam_args_path = os.path.join(cfg.VCAM_YOUTUBE_FINETUNE_RUNS_DIR, 
                                   cfg.VCAM_YOUTUBE_FINETUNE_MODEL_NAME, 'args.yaml')
    vcam_settings = {
        'Batch Size': cfg.VCAM_YOUTUBE_FINETUNE_BATCH_SIZE,
        'Learning Rate (lr0)': cfg.VCAM_YOUTUBE_FINETUNE_LR0,
        'Learning Rate Final (lrf)': cfg.VCAM_YOUTUBE_FINETUNE_LRF,
        'Epochs': cfg.VCAM_YOUTUBE_FINETUNE_EPOCHS,
        'Optimizer': 'AdamW (auto)',
        'Loss Function': 'YOLO Loss (box + cls + dfl)',
    }
    
    if os.path.exists(vcam_args_path):
        try:
            with open(vcam_args_path, 'r') as f:
                vcam_args = yaml.safe_load(f)
                if 'device' in vcam_args:
                    vcam_settings['Hardware'] = f"GPU {vcam_args['device']}" if vcam_args['device'] != 'cpu' else 'CPU'
        except:
            pass
    
    if 'Hardware' not in vcam_settings:
        vcam_settings['Hardware'] = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    settings['VCam Fine-tuned'] = vcam_settings
    
    # Audio Fine-tuned settings
    audio_settings = {
        'Batch Size': cfg.AUDIO_YOUTUBE_FINETUNE_BATCH_SIZE,
        'Learning Rate Stage 1': cfg.AUDIO_YOUTUBE_FINETUNE_LR_STAGE1,
        'Learning Rate Stage 2': cfg.AUDIO_YOUTUBE_FINETUNE_LR_STAGE2,
        'Epochs Stage 1': cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE1,
        'Epochs Stage 2': cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE2,
        'Optimizer': 'Adam',
        'Loss Function': 'Sparse Categorical Crossentropy',
        'Hardware': 'GPU' if torch.cuda.is_available() else 'CPU'
    }
    settings['Audio Fine-tuned'] = audio_settings
    
    # Early Fusion settings
    fusion_settings = {
        'Batch Size': cfg.FUSION_BATCH_SIZE,
        'Learning Rate Stage 1': cfg.FUSION_LEARNING_RATE_STAGE1,
        'Learning Rate Stage 2': cfg.FUSION_LEARNING_RATE_STAGE2,
        'Epochs Stage 1': cfg.FUSION_EPOCHS_STAGE1,
        'Epochs Stage 2': cfg.FUSION_EPOCHS_STAGE2,
        'Optimizer': 'Adam',
        'Loss Function': 'Sparse Categorical Crossentropy',
        'Hardware': 'GPU' if torch.cuda.is_available() else 'CPU'
    }
    settings['Early Fusion'] = fusion_settings
    
    # In ra console
    for model_name, params in settings.items():
        print(f"\n{model_name}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    return settings


def generate_comprehensive_report(results_dict, output_dir):
    """Tạo báo cáo tổng hợp"""
    os.makedirs(output_dir, exist_ok=True)
    
    # TABLE I
    table_i_path = os.path.join(output_dir, 'TABLE_I_Overall_Comparison_OOD.csv')
    table_i_data, df_i = generate_table_i(results_dict, table_i_path)
    
    # TABLE II
    table_ii_path = os.path.join(output_dir, 'TABLE_II_Per_Class_Performance.csv')
    table_ii_data, df_ii = generate_table_ii(results_dict, table_ii_path)
    
    # Confusion Matrices
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    generate_confusion_matrices(results_dict, cm_dir)
    
    # Training Settings
    training_settings = collect_training_settings()
    settings_path = os.path.join(output_dir, 'training_settings.txt')
    with open(settings_path, 'w', encoding='utf-8') as f:
        f.write("TRAINING SETTINGS\n")
        f.write("="*80 + "\n\n")
        for model_name, params in training_settings.items():
            f.write(f"{model_name}:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    print(f"\nTraining Settings đã lưu: {settings_path}")
    
    # Tạo báo cáo Markdown tổng hợp
    report_path = os.path.join(output_dir, 'comprehensive_evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Evaluation Report\n\n")
        
        f.write("## TABLE I: Overall Comparison on OOD\n\n")
        f.write(df_i.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## TABLE II: Per-Class Performance\n\n")
        f.write(df_ii.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Confusion Matrices\n\n")
        f.write("### VCam Fine-tuned\n")
        f.write("![VCam CM](confusion_matrices/confusion_matrix_vcam.png)\n\n")
        f.write("### Audio Fine-tuned\n")
        f.write("![Audio CM](confusion_matrices/confusion_matrix_audio.png)\n\n")
        f.write("### Early Fusion\n")
        f.write("![Fusion CM](confusion_matrices/confusion_matrix_fusion.png)\n\n")
        
        f.write("## Training Settings\n\n")
        for model_name, params in training_settings.items():
            f.write(f"### {model_name}\n\n")
            for key, value in params.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
    
    print(f"\nBáo cáo tổng hợp đã lưu: {report_path}")


def main():
    """Hàm chính"""
    print("="*80)
    print("TẠO BÁO CÁO ĐÁNH GIÁ TỔNG HỢP")
    print("="*80)
    
    # Đánh giá trên OOD
    results = evaluate_all_models_on_ood_for_report()
    
    if results is None:
        print("\nLỖI: Không thể đánh giá models. Vui lòng kiểm tra lại.")
        return
    
    # Tạo báo cáo
    output_dir = os.path.join(PROJECT_ROOT, 'reports', 'comprehensive_evaluation')
    generate_comprehensive_report(results, output_dir)
    
    print("\n" + "="*80)
    print("HOÀN TẤT!")
    print("="*80)
    print(f"Kết quả đã lưu tại: {output_dir}")


if __name__ == "__main__":
    main()
