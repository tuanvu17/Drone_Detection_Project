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
import cv2

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

def get_yolo_classification_predictions_and_visualize(model, data_yaml_path, split='test',
                                                     output_annotated_dir=None, conf_thresh_viz=0.4):
    print(f"Bắt đầu lấy dự đoán lớp và trực quan hóa cho tập '{split}' của {data_yaml_path}")
    data_yaml_dir = os.path.dirname(data_yaml_path)
    test_images_dir = os.path.join(data_yaml_dir, split, 'images')

    if not os.path.exists(test_images_dir):
        print(f"LỖI: Không tìm thấy thư mục ảnh test tại: {test_images_dir}")
        return [], [], []

    if output_annotated_dir:
        os.makedirs(output_annotated_dir, exist_ok=True)
        print(f"Ảnh có bounding box sẽ được lưu vào: {output_annotated_dir}")

    results_list = model.predict(source=test_images_dir,
                                 imgsz=cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                                 conf=0.25, # Ngưỡng conf cho việc lấy dự đoán lớp (có thể thấp hơn viz)
                                 stream=False,
                                 verbose=False)

    true_labels_encoded_for_report = []
    pred_labels_encoded_for_report = []

    # Lấy tên lớp từ model YOLO đã tải
    if isinstance(model.names, dict):
        class_names_from_model = [model.names[i] for i in sorted(model.names.keys())]
    elif isinstance(model.names, list):
        class_names_from_model = model.names
    else:
        print("LỖI: Không thể xác định class_names_from_model từ model.names.")
        return [], [], []

    # Tạo LabelEncoder DỰA TRÊN CÁC LỚP CỦA MODEL YOLO
    # Điều này quan trọng để đảm bảo ID lớp khớp giữa ground truth và dự đoán
    le_yolo_model_classes = LabelEncoder()
    le_yolo_model_classes.fit(class_names_from_model)


    for i, res in enumerate(results_list):
        img_path = res.path
        img_filename = os.path.basename(img_path)
        label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', os.path.splitext(img_filename)[0] + '.txt')
        
        frame_to_draw = cv2.imread(img_path)
        if frame_to_draw is None:
            print(f"Cảnh báo: Không thể đọc ảnh {img_path}. Bỏ qua ảnh này.")
            continue
        
        true_class_id_yolo_for_report = -1
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f_label:
                    first_line = f_label.readline().strip()
                    if first_line: # Đảm bảo dòng không rỗng
                        true_class_id_yolo_for_report = int(first_line.split()[0])
                        # Kiểm tra xem class ID này có nằm trong phạm vi các lớp của model không
                        if true_class_id_yolo_for_report >= len(class_names_from_model):
                            print(f"Cảnh báo: Ground truth class ID {true_class_id_yolo_for_report} cho {img_filename} nằm ngoài phạm vi các lớp của model ({len(class_names_from_model)} lớp). Bỏ qua.")
                            true_class_id_yolo_for_report = -1 # Đặt lại để bỏ qua
            except ValueError:
                 print(f"Cảnh báo: Không thể chuyển đổi class ID thành số nguyên từ file {label_path}. Dòng: '{first_line}'. Bỏ qua.")
                 true_class_id_yolo_for_report = -1
            except Exception as e:
                print(f"Cảnh báo: Lỗi khi đọc nhãn từ {label_path}: {e}. Bỏ qua.")
                true_class_id_yolo_for_report = -1
        else:
            print(f"Cảnh báo: Không tìm thấy file nhãn {label_path} cho ảnh {img_filename}. Bỏ qua ảnh này cho report.")
            # Nếu không có file nhãn, không thể đưa vào report, nhưng vẫn có thể vẽ dự đoán nếu có
        
        # Nếu không có ground truth hợp lệ, không thêm vào report, nhưng vẫn có thể vẽ dự đoán
        if true_class_id_yolo_for_report == -1 and output_annotated_dir:
            print(f"  Ảnh {img_filename} sẽ không được tính vào report do thiếu GT hợp lệ, nhưng sẽ thử vẽ dự đoán.")


        pred_class_id_yolo_for_report = -1
        max_conf_for_report = 0.0
        boxes_to_draw = []

        if res.boxes and hasattr(res.boxes, 'conf') and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            for box_idx in range(len(res.boxes.conf)):
                conf = float(res.boxes.conf[box_idx])
                cls_id = int(res.boxes.cls[box_idx])

                if cls_id >= len(class_names_from_model): # Bỏ qua nếu ID lớp dự đoán không hợp lệ
                    print(f"Cảnh báo: Dự đoán class ID {cls_id} cho {img_filename} nằm ngoài phạm vi các lớp của model. Bỏ qua dự đoán này.")
                    continue

                if conf > max_conf_for_report:
                    max_conf_for_report = conf
                    pred_class_id_yolo_for_report = cls_id
                
                if conf >= conf_thresh_viz:
                    coords = res.boxes.xyxy[box_idx].cpu().numpy().astype(int)
                    class_name_viz = class_names_from_model[cls_id]
                    text_to_draw = f"{class_name_viz}: {conf:.2f}"
                    boxes_to_draw.append({'coords': coords, 'text': text_to_draw, 'class_id': cls_id, 'class_name': class_name_viz})
        
        # Chỉ thêm vào report nếu có ground truth hợp lệ
        if true_class_id_yolo_for_report != -1:
            if pred_class_id_yolo_for_report != -1:
                true_labels_encoded_for_report.append(true_class_id_yolo_for_report)
                pred_labels_encoded_for_report.append(pred_class_id_yolo_for_report)
            else: # GT có, model không detect gì -> FN
                true_labels_encoded_for_report.append(true_class_id_yolo_for_report)
                dummy_pred_id = (true_class_id_yolo_for_report + 1) % len(class_names_from_model)
                pred_labels_encoded_for_report.append(dummy_pred_id)
                # print(f"  Ảnh {img_filename}: GT là '{class_names_from_model[true_class_id_yolo_for_report]}', YOLO không detect. Ghi nhận là FN cho report.")

        if output_annotated_dir:
            drawn_image = frame_to_draw.copy() # Làm việc trên bản sao
            if boxes_to_draw:
                for box_info in boxes_to_draw:
                    x1, y1, x2, y2 = box_info['coords']
                    text = box_info['text']
                    class_name_iter = box_info['class_name'] # Sử dụng class_name đã lấy được

                    color = (0, 255, 0) # Xanh lá mặc định
                    if class_name_iter == 'DRONE': color = (0, 255, 0)
                    elif class_name_iter == 'HELICOPTER': color = (0, 0, 255) # Đỏ
                    elif class_name_iter == 'AIRPLANE': color = (255, 0, 0) # Xanh dương
                    elif class_name_iter == 'BIRD': color = (0, 255, 255) # Vàng
                    # Thêm màu cho BACKGROUND nếu nó có trong model.names và bạn muốn vẽ
                    # elif class_name_iter == 'BACKGROUND': color = (128, 128, 128) # Xám

                    cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(drawn_image, text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                save_path = os.path.join(output_annotated_dir, f"annotated_{img_filename}")
                cv2.imwrite(save_path, drawn_image)
            else: # Không có box nào đạt ngưỡng conf_thresh_viz
                save_path = os.path.join(output_annotated_dir, f"no_high_conf_detection_{img_filename}")
                cv2.imwrite(save_path, drawn_image) # Lưu ảnh gốc nếu không có detection đáng kể

    if not true_labels_encoded_for_report or not pred_labels_encoded_for_report:
        print("Không có đủ dữ liệu (true_labels hoặc pred_labels) để tạo báo cáo phân loại.")
        return np.array([]), np.array([]), class_names_from_model # Trả về class_names_from_model để tránh lỗi sau

    return np.array(true_labels_encoded_for_report), np.array(pred_labels_encoded_for_report), class_names_from_model


def run_vcam_yolo_evaluation_on_test_in_domain():
    print("===== EVALUATING FINE-TUNED VCAM YOLO MODEL ON ITS YOUTUBE TEST SET =====")
    if hasattr(cfg_eval_vcam, 'ensure_output_directories'): # Gọi hàm ensure nếu nó tồn tại trong cfg
        cfg_eval_vcam.ensure_output_directories()
    else: # Nếu không, tự tạo các thư mục cần thiết cho script này
        os.makedirs(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR, exist_ok=True)
        os.makedirs(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR, exist_ok=True)
        if hasattr(cfg_eval_vcam, 'VCAM_YOUTUBE_FINETUNE_ANNOTATED_TEST_IMAGES_DIR'):
            os.makedirs(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_ANNOTATED_TEST_IMAGES_DIR, exist_ok=True)


    model_path = cfg_eval_vcam.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH
    data_yaml_path = cfg_eval_vcam.VCAM_YOUTUBE_DATA_YAML_PATH

    # ... (Phần kiểm tra file tồn tại giữ nguyên) ...
    if not os.path.exists(model_path): # ...
        return
    if not os.path.exists(data_yaml_path): # ...
        return
    data_yaml_dir = os.path.dirname(data_yaml_path)
    test_images_path_check = os.path.join(data_yaml_dir, 'test', 'images')
    if not os.path.exists(test_images_path_check) or not os.listdir(test_images_path_check): # ...
        return

    print(f"Đang tải mô hình VCam đã fine-tune từ: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e: # ...
        return

    # --- Chạy model.val() để lấy các chỉ số detection (mAP) ---
    print("\n--- Chạy model.val() để lấy các chỉ số detection (mAP) ---")
    try:
        val_results = model.val(data=data_yaml_path,
                                split="test",
                                imgsz=cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
                                batch=max(1, cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_BATCH_SIZE // 2),
                                device='0' if torch.cuda.is_available() else 'cpu',
                                save_json=True,
                                save_hybrid=False, # Tắt để tránh cảnh báo và đảm bảo mAP chính xác
                                name=f"{cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_MODEL_NAME}_eval_on_yt_test_mAP_only"
                                )
        print("\nKết quả đánh giá Detection từ model.val():")
        if hasattr(val_results, 'box') and val_results.box is not None:
            print(f"  mAP50-95: {val_results.box.map:.4f}")
            print(f"  mAP50: {val_results.box.map50:.4f}")
            print(f"  mAP75: {val_results.box.map75:.4f}")
            
            map_metrics_path = os.path.join(cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR,
                                           f"vcam_yt_finetuned_map_metrics_on_test.txt")
            with open(map_metrics_path, 'w') as f:
                f.write("VCam YouTube Fine-tuned Model - Detection Metrics on its Test Set\n")
                f.write("="*60 + "\n")
                f.write(f"mAP50-95: {val_results.box.map:.4f}\n")
                f.write(f"mAP50: {val_results.box.map50:.4f}\n")
                f.write(f"mAP75: {val_results.box.map75:.4f}\n")
                # Ghi mAP per class (nếu có)
                if hasattr(val_results.box, 'maps') and val_results.box.maps is not None: # maps là mAP50-95 cho từng lớp
                    f.write("mAP50-95 per class:\n")
                    current_model_names = model.names
                    if isinstance(current_model_names, dict): # Sắp xếp theo ID
                        for class_id_idx in sorted(current_model_names.keys()):
                            class_name_val = current_model_names[class_id_idx]
                            if class_id_idx < len(val_results.box.maps):
                                f.write(f"  - {class_name_val} (ID {class_id_idx}): {val_results.box.maps[class_id_idx]:.4f}\n")
                    elif isinstance(current_model_names, list):
                         for class_id_idx, class_name_val in enumerate(current_model_names):
                            if class_id_idx < len(val_results.box.maps):
                                f.write(f"  - {class_name_val} (ID {class_id_idx}): {val_results.box.maps[class_id_idx]:.4f}\n")

            print(f"Đã lưu các chỉ số mAP vào: {map_metrics_path}")
        else:
            print("  Không có kết quả 'box' trong val_results để hiển thị mAP.")
    except Exception as e: # ...
        print(f"Lỗi trong quá trình chạy model.val(): {e}")
        import traceback
        traceback.print_exc()

    # --- Tạo Classification Report và Trực quan hóa Bounding Box ---
    print("\n--- Tạo Classification Report và Trực quan hóa Bounding Box ---")
    annotated_images_output_dir = cfg_eval_vcam.VCAM_YOUTUBE_FINETUNE_ANNOTATED_TEST_IMAGES_DIR
    
    true_labels_for_report, pred_labels_for_report, class_names_for_report_actual = \
        get_yolo_classification_predictions_and_visualize(
            model,
            data_yaml_path,
            split='test',
            output_annotated_dir=annotated_images_output_dir,
            conf_thresh_viz=0.4
        )

    # Kiểm tra lại sau khi gọi hàm
    if isinstance(true_labels_for_report, np.ndarray) and true_labels_for_report.size > 0 and \
       isinstance(pred_labels_for_report, np.ndarray) and pred_labels_for_report.size > 0:
        if not class_names_for_report_actual or not isinstance(class_names_for_report_actual, list) or not class_names_for_report_actual:
            print("Lỗi: class_names_for_report_actual không hợp lệ sau khi gọi hàm get_yolo. Không thể tạo báo cáo.")
        else:
            report_str = classification_report(
                true_labels_for_report,
                pred_labels_for_report,
                target_names=class_names_for_report_actual,
                labels=list(range(len(class_names_for_report_actual))), # Dùng các ID lớp thực tế từ model
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
        print("Không có đủ dữ liệu (true_labels_for_report hoặc pred_labels_for_report rỗng) để tạo Classification Report hoặc Confusion Matrix sau khi trực quan hóa.")

    print("===== EVALUATION OF FINE-TUNED VCAM YOLO MODEL FINISHED =====")

# if __name__ == '__main__':
#     run_vcam_yolo_evaluation_on_test_in_domain()