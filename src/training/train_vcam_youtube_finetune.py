# Drone_Detection_Project/src/training/train_vcam_youtube_finetune.py
import os
import shutil
from ultralytics import YOLO
import torch
import yaml # Để đọc file YAML và lấy danh sách lớp

# Import config
try:
    from src.config_loader.loader import get_config
    cfg = get_config()
except ImportError:
    import sys
    PROJECT_ROOT_FT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT_FT not in sys.path:
        sys.path.append(PROJECT_ROOT_FT)
    from config import project_config as cfg

def run_vcam_youtube_finetuning():
    print("===== STARTING VCAM YOLO MODEL FINE-TUNING ON YOUTUBE DATA =====")
    if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories):
        print("Đảm bảo các thư mục output tồn tại...")
        cfg.ensure_output_directories()
    else:
        print("Cảnh báo: Hàm ensure_output_directories không khả dụng. Tự tạo thư mục nếu cần.")
        os.makedirs(cfg.VCAM_YOUTUBE_FINETUNE_RUNS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH), exist_ok=True)
        # Các thư mục reports sẽ được tạo bởi các hàm lưu kết quả nếu chúng chưa tồn tại
        if hasattr(cfg, 'VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR'):
            os.makedirs(cfg.VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR, exist_ok=True)
        if hasattr(cfg, 'VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR'):
            os.makedirs(cfg.VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR, exist_ok=True)


    # 1. Kiểm tra các file đầu vào
    if not os.path.exists(cfg.VCAM_YOUTUBE_DATA_YAML_PATH):
        print(f"LỖI: Không tìm thấy file data.yaml cho VCam YouTube fine-tune tại: {cfg.VCAM_YOUTUBE_DATA_YAML_PATH}")
        return

    if not os.path.exists(cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH): # Model VCam gốc
        print(f"LỖI: Không tìm thấy mô hình VCam gốc để fine-tune tại: {cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
        return

    # Đọc class names từ file YAML để sử dụng sau này cho báo cáo
    class_names_from_yaml = []
    try:
        with open(cfg.VCAM_YOUTUBE_DATA_YAML_PATH, 'r') as f_yaml:
            data_yaml_content = yaml.safe_load(f_yaml)
            if 'names' in data_yaml_content and isinstance(data_yaml_content['names'], list):
                class_names_from_yaml = data_yaml_content['names']
                print(f"Các lớp được định nghĩa trong {cfg.VCAM_YOUTUBE_DATA_YAML_PATH}: {class_names_from_yaml}")
            else:
                print(f"CẢNH BÁO: Không tìm thấy 'names' hoặc 'names' không phải là list trong {cfg.VCAM_YOUTUBE_DATA_YAML_PATH}.")
    except Exception as e_yaml:
        print(f"Lỗi khi đọc file YAML: {e_yaml}")
    
    # Sử dụng MASTER_CLASS_LIST_FUSION nếu class_names_from_yaml rỗng hoặc có vấn đề
    # Hoặc bạn có thể quyết định dừng nếu YAML không hợp lệ.
    # Ở đây, chúng ta ưu tiên MASTER_CLASS_LIST_FUSION nếu muốn báo cáo đủ 5 lớp,
    # nhưng các chỉ số mAP từ YOLO.val() sẽ dựa trên class_names_from_yaml.
    # Để nhất quán, report nên dựa trên class_names_from_yaml mà model được train.
    if not class_names_from_yaml:
        print("Cảnh báo: Sử dụng cfg.CLASSES_YOLO (hoặc MASTER_CLASS_LIST_FUSION) làm danh sách lớp mặc định do lỗi đọc YAML.")
        class_names_for_report = cfg.CLASSES_YOLO # Hoặc cfg.MASTER_CLASS_LIST_FUSION nếu bạn muốn
    else:
        class_names_for_report = class_names_from_yaml


    # 2. Tải mô hình VCam gốc để fine-tune
    print(f"Loading VCam base model to fine-tune from: {cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
    model = YOLO(cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH)

    device_to_use = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_to_use}")

    # 3. Huấn luyện (Fine-tuning)
    print(f"Starting fine-tuning for VCam model on YouTube data (experiment name: {cfg.VCAM_YOUTUBE_FINETUNE_MODEL_NAME})")
    training_results = model.train(
        data=cfg.VCAM_YOUTUBE_DATA_YAML_PATH,
        epochs=cfg.VCAM_YOUTUBE_FINETUNE_EPOCHS,
        imgsz=cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
        batch=cfg.VCAM_YOUTUBE_FINETUNE_BATCH_SIZE,
        name=cfg.VCAM_YOUTUBE_FINETUNE_MODEL_NAME, # Tên thư mục con trong project
        project=cfg.VCAM_YOUTUBE_FINETUNE_RUNS_DIR, # Thư mục gốc cho tất cả các lần chạy fine-tune VCam YT
        exist_ok=True,
        device=device_to_use,
        verbose=True,
        lr0=cfg.VCAM_YOUTUBE_FINETUNE_LR0,
        lrf=cfg.VCAM_YOUTUBE_FINETUNE_LRF,
        warmup_epochs= getattr(cfg, 'VCAM_YOUTUBE_FINETUNE_WARMUP_EPOCHS', 3),
        patience=getattr(cfg, 'VCAM_YOUTUBE_FINETUNE_PATIENCE', 20),
        augment=True,
        cos_lr=True,
        weight_decay=getattr(cfg, 'VCAM_YOUTUBE_FINETUNE_WEIGHT_DECAY', 0.0005),
        # Augmentation params
        mixup=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('mixup', 0.0),
        mosaic=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('mosaic', 0.8),
        copy_paste=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('copy_paste', 0.0),
        hsv_h=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('hsv_h', 0.015),
        hsv_s=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('hsv_s', 0.7),
        hsv_v=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('hsv_v', 0.4),
        degrees=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('degrees', 0.0),
        translate=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('translate', 0.1),
        scale=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('scale', 0.1),
        fliplr=cfg.VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS.get('fliplr', 0.5),
        cache=True # hoặc 'ram'
    )
    print("--- VCam YouTube Data Fine-tuning Finished ---")

    # 4. Lưu mô hình tốt nhất
    yolo_finetuned_best_model_path_in_runs = model.trainer.best # Đường dẫn đến best.pt trong thư mục runs/train/<name>
    if os.path.exists(yolo_finetuned_best_model_path_in_runs):
        print(f"Copying best fine-tuned VCam model from {yolo_finetuned_best_model_path_in_runs} to {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH}")
        shutil.copy2(yolo_finetuned_best_model_path_in_runs, cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        print(f"VCam model fine-tuned on YouTube data saved to: {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH}")
    else:
        print(f"LỖI: Không tìm thấy best.pt sau khi fine-tune tại {yolo_finetuned_best_model_path_in_runs}")
        print("Kiểm tra lại thư mục runs hoặc quá trình huấn luyện có thể đã không tạo ra best.pt.")
        # return # Có thể dừng nếu không có model tốt nhất để đánh giá

    # 5. Đánh giá mô hình đã fine-tune trên tập test của bộ YouTube
    path_to_yaml_dir = os.path.dirname(cfg.VCAM_YOUTUBE_DATA_YAML_PATH)
    # Đường dẫn trong file yaml là tương đối với thư mục chứa file yaml, hoặc tuyệt đối
    # Giả sử 'test: test/images' trong yaml
    test_images_dir_for_eval_check = os.path.join(path_to_yaml_dir, data_yaml_content.get('test', 'test/images'))
    # Nếu đường dẫn trong yaml là tuyệt đối, không cần os.path.join
    if not os.path.isabs(data_yaml_content.get('test', '')):
         test_images_dir_for_eval_check = os.path.join(path_to_yaml_dir, data_yaml_content.get('test', 'test/images'))
    else:
         test_images_dir_for_eval_check = data_yaml_content.get('test', 'test/images')


    if os.path.exists(test_images_dir_for_eval_check) and os.listdir(test_images_dir_for_eval_check):
        print("\n--- Evaluating Fine-tuned VCam Model on its YouTube Test Set ---")
        # Tải lại mô hình tốt nhất đã lưu vào thư mục models của bạn để đảm bảo
        if not os.path.exists(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH):
            print(f"LỖI: Mô hình fine-tuned tại {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH} không tồn tại để đánh giá.")
            return
            
        model_to_eval = YOLO(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        
        # Thư mục lưu kết quả và biểu đồ của lần đánh giá này
        # Sẽ nằm trong VCAM_YOUTUBE_FINETUNE_RUNS_DIR/<VCAM_YOUTUBE_FINETUNE_MODEL_NAME>_evaluation_on_its_test/
        eval_run_name = f"{cfg.VCAM_YOUTUBE_FINETUNE_MODEL_NAME}_evaluation_on_its_test"
        
        # Tạo thư mục figures và metrics riêng cho lần đánh giá này trong reports
        eval_figures_dir = os.path.join(cfg.VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR, eval_run_name)
        eval_metrics_dir = os.path.join(cfg.VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR, eval_run_name)
        os.makedirs(eval_figures_dir, exist_ok=True)
        os.makedirs(eval_metrics_dir, exist_ok=True)

        print(f"Kết quả đánh giá chi tiết (bao gồm biểu đồ) sẽ được YOLO lưu vào thư mục con của: {cfg.VCAM_YOUTUBE_FINETUNE_RUNS_DIR}")
        print(f"Chúng tôi sẽ cố gắng sao chép các biểu đồ quan trọng sang: {eval_figures_dir}")


        test_metrics_results = model_to_eval.val(
            data=cfg.VCAM_YOUTUBE_DATA_YAML_PATH,
            split="test", # Yêu cầu YOLO sử dụng tập test được định nghĩa trong YAML
            device=device_to_use,
            batch=max(1, cfg.VCAM_YOUTUBE_FINETUNE_BATCH_SIZE // 2),
            imgsz=cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE,
            project=cfg.VCAM_YOUTUBE_FINETUNE_RUNS_DIR, # Nơi YOLO lưu output của .val()
            name=eval_name,      # Tên thư mục con cụ thể cho lần chạy .val() này
            save_json=True,      # Lưu kết quả ở định dạng COCO JSON
            save_txt=True,       # Lưu kết quả ở định dạng YOLO txt
            plots=True           # Yêu cầu YOLO tự tạo và lưu các biểu đồ (P-R, Confusion Matrix, ...)
        )
        
        # Đường dẫn đến thư mục mà YOLO đã lưu kết quả của lệnh .val()
        yolo_val_output_dir = os.path.join(cfg.VCAM_YOUTUBE_FINETUNE_RUNS_DIR, eval_name)
        print(f"\nTest set evaluation results (from .val() command, output in {yolo_val_output_dir}):")

        if hasattr(test_metrics_results, 'box') and test_metrics_results.box is not None:
            map50_95 = test_metrics_results.box.map
            map50 = test_metrics_results.box.map50
            map75 = test_metrics_results.box.map75
            maps_per_class = test_metrics_results.box.maps # mAP50-95 cho từng lớp (array)

            print(f"  mAP50-95 (all classes): {map50_95:.4f}")
            print(f"  mAP50 (all classes): {map50:.4f}")
            print(f"  mAP75 (all classes): {map75:.4f}")

            metrics_summary_path = os.path.join(eval_metrics_dir, f"detection_metrics_summary.txt")
            with open(metrics_summary_path, 'w') as f:
                f.write(f"Evaluation Metrics for {cfg.VCAM_YOUTUBE_FINETUNE_MODEL_NAME} on its YouTube Test Set (Detection)\n")
                f.write("="*80 + "\n")
                f.write(f"mAP@0.50-0.95 (All Classes): {map50_95:.4f}\n")
                f.write(f"mAP@0.50 (All Classes): {map50:.4f}\n")
                f.write(f"mAP@0.75 (All Classes): {map75:.4f}\n\n")
                
                f.write("mAP@0.50-0.95 per class (based on model.names order):\n")
                # Sử dụng class_names_for_report đã đọc từ YAML
                # model_to_eval.names có thể là dict hoặc list, class_names_for_report là list
                # maps_per_class là một array, thứ tự của nó sẽ khớp với thứ tự ID lớp 0, 1, 2...
                
                # Tạo một dictionary để chứa mAP cho từng lớp có dữ liệu
                class_map_dict = {}
                if maps_per_class is not None and len(maps_per_class) > 0:
                    for i, class_name_in_yaml in enumerate(class_names_for_report):
                        if i < len(maps_per_class):
                            class_map_dict[class_name_in_yaml] = maps_per_class[i]
                            f.write(f"  - {class_name_in_yaml} (ID {i}): {maps_per_class[i]:.4f}\n")
                        else: # Trường hợp maps_per_class ngắn hơn class_names_for_report (ít xảy ra)
                            f.write(f"  - {class_name_in_yaml} (ID {i}): N/A (no data in maps_per_class array)\n")
                
                # In ra cho 5 lớp bạn muốn, kể cả khi mAP=0
                # Giả sử MASTER_CLASS_LIST_FUSION là ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER', 'BACKGROUND']
                # và class_names_for_report là ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
                f.write("\nReporting for specified master classes (including those with no test data):\n")
                for master_class_name in cfg.MASTER_CLASS_LIST_FUSION:
                    # Nếu master_class_name có trong class_map_dict (tức là có trong YAML và có kết quả mAP)
                    if master_class_name in class_map_dict:
                        f.write(f"  - {master_class_name}: {class_map_dict[master_class_name]:.4f}\n")
                    # Nếu master_class_name không có trong YAML (ví dụ BACKGROUND) hoặc không có dữ liệu test
                    else:
                        # BACKGROUND sẽ không có mAP từ YOLO detection vì nó không phải là lớp đối tượng được detect
                        if master_class_name == "BACKGROUND":
                             f.write(f"  - {master_class_name}: N/A (not an object class for YOLO mAP)\n")
                        else: # Các lớp như AIRPLANE, BIRD nếu không có trong YAML hoặc không có GT trong test
                             f.write(f"  - {master_class_name}: 0.0000 (or N/A if not in YAML's 'names')\n")
            print(f"Test detection metrics summary saved to: {metrics_summary_path}")

            # Copy các biểu đồ quan trọng từ thư mục runs của YOLO sang thư mục reports của bạn
            yolo_plot_files = {
                "confusion_matrix.png": "detection_confusion_matrix.png",
                "P_curve.png": "detection_P_curve.png",
                "R_curve.png": "detection_R_curve.png",
                "PR_curve.png": "detection_PR_curve.png",
                # "F1_curve.png": "detection_F1_curve.png", # Tùy phiên bản YOLO có file này không
                # "labels.jpg": "test_batch_labels.jpg",
                # "val_batch0_pred.jpg": "test_batch0_pred_example.jpg" # Ví dụ dự đoán trên batch đầu tiên
            }
            for yolo_filename, report_filename in yolo_plot_files.items():
                src_plot_path = os.path.join(yolo_val_output_dir, yolo_filename)
                dest_plot_path = os.path.join(eval_figures_dir, report_filename)
                if os.path.exists(src_plot_path):
                    try:
                        shutil.copy2(src_plot_path, dest_plot_path)
                        print(f"  Copied {yolo_filename} to {dest_plot_path}")
                    except Exception as e_copy:
                        print(f"  Lỗi khi copy {src_plot_path}: {e_copy}")
                # else:
                    # print(f"  Không tìm thấy file biểu đồ của YOLO: {src_plot_path}")

        else:
            print("  Không có kết quả 'box' trong test_metrics_results để hiển thị mAP chi tiết.")
    else:
        print(f"Không tìm thấy thư mục test ({test_images_dir_for_eval_check}) hoặc thư mục rỗng.")

    print("===== VCAM YOLO MODEL FINE-TUNING ON YOUTUBE DATA PIPELINE FINISHED =====")

if __name__ == '__main__':
    run_vcam_youtube_finetuning()