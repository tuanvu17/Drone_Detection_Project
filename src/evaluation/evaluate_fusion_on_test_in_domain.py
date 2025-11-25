# Drone_Detection_Project/src/evaluation/evaluate_vcam_yolo_on_test_in_domain.py
import os
import shutil
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix as sklearn_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 # <<< THÊM IMPORT CV2

# Import config và các hàm tiện ích
try:
    from src.config_loader.loader import get_config
    from src.evaluation.plotting_utils import plot_custom_confusion_matrix
except ImportError:
    import sys
    current_dir_eval_vcam = os.path.dirname(os.path.abspath(__file__))
    src_dir_eval_vcam = os.path.dirname(current_dir_eval_vcam)
    project_root_eval_vcam = os.path.dirname(src_dir_eval_vcam)
    if project_root_eval_vcam not in sys.path: sys.path.insert(0, project_root_eval_vcam)
    if src_dir_eval_vcam not in sys.path: sys.path.insert(0, src_dir_eval_vcam)
    from config_loader.loader import get_config
    from evaluation.plotting_utils import plot_custom_confusion_matrix

cfg_eval_vcam = get_config()

# --- Hàm lấy dự đoán lớp VÀ VẼ BOUNDING BOX ---
def get_yolo_classification_predictions_and_visualize(model, data_yaml_path, split='test',
                                                     output_annotated_dir=None, conf_thresh_viz=0.4): # Thêm output_dir và ngưỡng conf cho visualize
    """
    Chạy model.predict() trên tập test, trích xuất nhãn dự đoán và nhãn thực tế
    cho classification report, đồng thời vẽ bounding box và confidence lên ảnh và lưu lại.
    """
    print(f"Bắt đầu lấy dự đoán lớp và trực quan hóa cho tập '{split}' của {data_yaml_path}")
    data_yaml_dir = os.path.dirname(data_yaml_path)
    test_images_dir = os.path.join(data_yaml_dir, split, 'images')

    if not os.path.exists(test_images_dir):
        print(f"LỖI: Không tìm thấy thư mục ảnh test tại: {test_images_dir}")
        return [], [], []

    if output_annotated_dir:
        os.makedirs(output_annotated_dir, exist_ok=True)
        print(f"Ảnh có bounding box sẽ được lưu vào: {output_annotated_dir}")

    # Chạy predict một lần
    results_list = model.predict(source=test_images_dir,
                                 imgsz=cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                                 conf=0.25, # Ngưỡng conf thấp hơn cho việc lấy dự đoán lớp (nếu cần)
                                 stream=False,
                                 verbose=False)

    true_labels_encoded_for_report = []
    pred_labels_encoded_for_report = []

    if isinstance(model.names, dict):
        class_names_from_model = [model.names[i] for i in sorted(model.names.keys())]
    else:
        class_names_from_model = model.names

    le = LabelEncoder()
    le.fit(class_names_from_model)

    for i, res in enumerate(results_list):
        img_path = res.path
        img_filename = os.path.basename(img_path)
        label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', os.path.splitext(img_filename)[0] + '.txt')

        # Đọc ảnh gốc để vẽ
        frame_to_draw = cv2.imread(img_path)
        if frame_to_draw is None:
            print(f"Cảnh báo: Không thể đọc ảnh {img_path} để vẽ. Bỏ qua trực quan hóa cho ảnh này.")
            # Vẫn tiếp tục xử lý để lấy label cho report nếu có thể
        
        # --- Lấy Ground Truth cho Classification Report ---
        true_class_id_yolo_for_report = -1
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f_label:
                    line = f_label.readline().strip()
                    if line:
                        true_class_id_yolo_for_report = int(line.split()[0])
            except Exception as e:
                print(f"Cảnh báo: Không thể đọc nhãn từ {label_path}: {e}")
        else:
            print(f"Cảnh báo: Không tìm thấy file nhãn {label_path} cho ảnh {img_path}")

        if true_class_id_yolo_for_report == -1:
            print(f"  Bỏ qua ảnh {img_filename} cho classification report do thiếu nhãn ground truth.")
            # Nếu muốn lưu ảnh không có GT box nhưng có prediction box thì vẫn tiếp tục phần vẽ
        
        # --- Lấy Dự đoán lớp cho Classification Report (đối tượng có confidence cao nhất) ---
        pred_class_id_yolo_for_report = -1
        max_conf_for_report = 0.0
        
        # Biến để lưu box và text sẽ vẽ (chỉ những box > conf_thresh_viz)
        boxes_to_draw = []

        if res.boxes and len(res.boxes.conf) > 0:
            for box_idx in range(len(res.boxes.conf)):
                conf = float(res.boxes.conf[box_idx])
                cls_id = int(res.boxes.cls[box_idx])
                
                # Cập nhật dự đoán cho report (luôn lấy đối tượng có conf cao nhất)
                if conf > max_conf_for_report:
                    max_conf_for_report = conf
                    pred_class_id_yolo_for_report = cls_id

                # Chỉ thêm vào danh sách vẽ nếu confidence > ngưỡng visualize
                if conf >= conf_thresh_viz and frame_to_draw is not None:
                    coords = res.boxes.xyxy[box_idx].cpu().numpy().astype(int) # x1, y1, x2, y2
                    class_name = class_names_from_model[cls_id]
                    text_to_draw = f"{class_name}: {conf:.2f}"
                    boxes_to_draw.append({'coords': coords, 'text': text_to_draw, 'class_id': cls_id})
        
        # --- Thêm vào danh sách cho classification report ---
        if true_class_id_yolo_for_report != -1 : # Chỉ thêm vào report nếu có GT
            if pred_class_id_yolo_for_report != -1: # Nếu có dự đoán
                true_labels_encoded_for_report.append(true_class_id_yolo_for_report)
                pred_labels_encoded_for_report.append(pred_class_id_yolo_for_report)
            else: # GT có, nhưng model không detect gì -> False Negative
                true_labels_encoded_for_report.append(true_class_id_yolo_for_report)
                dummy_pred_id = (true_class_id_yolo_for_report + 1) % len(class_names_from_model)
                pred_labels_encoded_for_report.append(dummy_pred_id)
                print(f"  Ảnh {img_filename}: GT là '{class_names_from_model[true_class_id_yolo_for_report]}', YOLO không detect (cho report).")

        # --- Vẽ Bounding Box và Confidence lên ảnh ---
        if frame_to_draw is not None and output_annotated_dir and boxes_to_draw:
            for box_info in boxes_to_draw:
                x1, y1, x2, y2 = box_info['coords']
                text = box_info['text']
                
                # Màu sắc có thể tùy chỉnh theo class_id
                # Ví dụ đơn giản: màu xanh cho DRONE, đỏ cho HELI
                color = (0, 255, 0) # Mặc định xanh lá
                if box_info['class_id'] == class_names_from_model.index('DRONE'): # Giả sử DRONE là một lớp
                     color = (0, 255, 0) # Xanh lá
                elif box_info['class_id'] == class_names_from_model.index('HELICOPTER'): # Giả sử HELICOPTER là một lớp
                     color = (0, 0, 255) # Đỏ

                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_to_draw, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            save_path = os.path.join(output_annotated_dir, f"annotated_{img_filename}")
            cv2.imwrite(save_path, frame_to_draw)
        elif frame_to_draw is not None and output_annotated_dir and not boxes_to_draw:
            # Nếu không có box nào để vẽ nhưng vẫn muốn lưu ảnh gốc (hoặc ảnh "không có phát hiện")
            save_path = os.path.join(output_annotated_dir, f"no_detection_{img_filename}")
            # cv2.putText(frame_to_draw, "No Detections", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Tùy chọn
            cv2.imwrite(save_path, frame_to_draw)


    if not true_labels_encoded_for_report or not pred_labels_encoded_for_report:
        print("Không có đủ dữ liệu nhãn thực tế hoặc dự đoán hợp lệ để tạo báo cáo phân loại.")
        return [], [], [] # Trả về list rỗng

    return np.array(true_labels_encoded_for_report), np.array(pred_labels_encoded_for_report), class_names_from_model


def run_fusion_evaluation_on_test_in_domain():
    print("===== EVALUATING FINE-TUNED VCAM YOLO MODEL ON ITS YOUTUBE TEST SET =====")
    if hasattr(cfg_eval_vcam, 'ensure_output_directories'):
        cfg_eval_vcam.ensure_output_directories()

    model_path = cfg_eval_vcam.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH
    data_yaml_path = cfg_eval_vcam.VCAM_YOUTUBE_DATA_YAML_PATH

    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy mô hình VCam đã fine-tune tại: {model_path}")
        return
    if not os.path.exists(data_yaml_path):
        print(f"LỖI: Không tìm thấy file data.yaml cho VCam YouTube tại: {data_yaml_path}")
        return

    data_yaml_dir = os.path.dirname(data_yaml_path)
    test_images_path_check = os.path.join(data_yaml_dir, 'test', 'images')
    if not os.path.exists(test_images_path_check) or not os.listdir(test_images_path_check):
        print(f"LỖI: Không tìm thấy thư mục test/images hoặc thư mục rỗng tại: {test_images_path_check}")
        return

    print(f"Đang tải mô hình VCam đã fine-tune từ: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi khi tải mô hình VCam: {e}")
        return

    # ... (Phần model.val() giữ nguyên như trước để lấy mAP) ...
    print("\n--- Chạy model.val() để lấy các chỉ số detection (mAP) ---")
    try:
        val_results = model.val(data=data_yaml_path,
                                split="test",
                                imgsz=cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                                batch=max(1, cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_BATCH_SIZE // 2),
                                device='0' if torch.cuda.is_available() else 'cpu',
                                save_json=True, # Vẫn nên giữ để có thể phân tích sâu
                                save_hybrid=False, # Tắt save_hybrid để tránh cảnh báo và đảm bảo mAP chính xác
                                name=f"{cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_MODEL_NAME}_eval_on_yt_test_mAP" # Đổi tên thư mục runs
                                )
        print("\nKết quả đánh giá Detection từ model.val():")
        if hasattr(val_results, 'box') and val_results.box is not None:
            print(f"  mAP50-95: {val_results.box.map:.4f}")
            print(f"  mAP50: {val_results.box.map50:.4f}")
            print(f"  mAP75: {val_results.box.map75:.4f}")
            
            map_metrics_path = os.path.join(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR,
                                           f"vcam_yt_finetuned_map_metrics_on_test.txt")
            os.makedirs(os.path.dirname(map_metrics_path), exist_ok=True)
            with open(map_metrics_path, 'w') as f:
                f.write("VCam YouTube Fine-tuned Model - Detection Metrics on its Test Set\n")
                f.write("="*60 + "\n")
                f.write(f"mAP50-95: {val_results.box.map:.4f}\n")
                f.write(f"mAP50: {val_results.box.map50:.4f}\n")
                f.write(f"mAP75: {val_results.box.map75:.4f}\n")
            print(f"Đã lưu các chỉ số mAP vào: {map_metrics_path}")
        else:
            print("  Không có kết quả 'box' trong val_results để hiển thị mAP.")

    except Exception as e:
        print(f"Lỗi trong quá trình chạy model.val(): {e}")
        import traceback
        traceback.print_exc()


    print("\n--- Tạo Classification Report và Trực quan hóa Bounding Box ---")
    # Sử dụng đường dẫn từ config cho thư mục lưu ảnh đã annotate
    annotated_images_output_dir = cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_ANNOTATED_TEST_IMAGES_DIR
    os.makedirs(annotated_images_output_dir, exist_ok=True) # Đảm bảo thư mục tồn tại

    true_labels_for_report, pred_labels_for_report, class_names_for_report_actual = \
        get_yolo_classification_predictions_and_visualize(
            model,
            data_yaml_path,
            split='test',
            output_annotated_dir=annotated_images_output_dir,
            conf_thresh_viz=0.4 # Ngưỡng confidence để vẽ box, bạn có thể điều chỉnh
        )

    if true_labels_for_report.size > 0 and pred_labels_for_report.size > 0:
        if not class_names_for_report_actual or not isinstance(class_names_for_report_actual, list):
            print("Lỗi: class_names_for_report_actual không hợp lệ. Không thể tạo báo cáo.")
        else:
            report_str = classification_report(
                true_labels_for_report,
                pred_labels_for_report,
                target_names=class_names_for_report_actual,
                labels=list(range(len(class_names_for_report_actual))),
                zero_division=0
            )
            print("\nClassification Report (VCam Fine-tuned on YouTube Test Set - Dựa trên dự đoán đối tượng chính):")
            print(report_str)

            report_file_path = os.path.join(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR,
                                            "vcam_yt_finetuned_classification_report_on_test.txt")
            with open(report_file_path, 'w') as f:
                f.write("Classification Report for VCam YouTube Fine-tuned Model on its Test Set (based on primary object)\n")
                f.write("="*70 + "\n")
                f.write(report_str)
            print(f"Đã lưu Classification Report vào: {report_file_path}")

            cm_filename = os.path.join(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR,
                                       "vcam_yt_finetuned_cm_on_test.png")
            plot_custom_confusion_matrix(true_labels_for_report, pred_labels_for_report,
                                         class_names_for_report_actual, filename=cm_filename,
                                         title="CM - VCam Fine-tuned (YouTube Test Set)")
    else:
        print("Không có đủ dữ liệu để tạo Classification Report hoặc Confusion Matrix sau khi trực quan hóa.")

    print("===== EVALUATION OF FINE-TUNED VCAM YOLO MODEL FINISHED =====")

if __name__ == '__main__':
    run_fusion_evaluation_on_test_in_domain()