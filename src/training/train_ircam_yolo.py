# Drone_Detection_Project/src/training/train_ircam_yolo.py
import os
import shutil
from ultralytics import YOLO
# Import config
try:
    from config import project_config as cfg
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from config import project_config as cfg

def run_ircam_yolo_training():
    print("===== STARTING IRCAM YOLO MODEL TRAINING =====")
    # Gọi hàm ensure_output_directories từ module cfg
    if hasattr(cfg, 'ensure_output_directories'):
        print("Ensuring output directories from train_ircam_yolo.py...")
        cfg.ensure_output_directories()
    else:
        print("Warning: ensure_output_directories function not found in config.")
    # Kiểm tra file data.yaml
    if not os.path.exists(cfg.IRCAM_DATA_YAML_PATH):
        print(f"Lỗi: Không tìm thấy file data.yaml cho IRCam tại: {cfg.IRCAM_DATA_YAML_PATH}")
        print("Vui lòng tạo file data.yaml với đường dẫn đến dữ liệu train/val/test cho IRCam.")
        print("Ví dụ nội dung data.yaml:")
        print(f"path: {os.path.dirname(cfg.IRCAM_DATA_YAML_PATH)}") # Thư mục chứa train, val, test
        print("train: train/images")
        print("val: val/images")
        print("test: test/images")
        print(f"nc: {cfg.NUM_CLASSES_YOLO}")
        print(f"names: {cfg.CLASSES_YOLO}")
        return

    # Tải pretrained model
    print(f"Loading pretrained YOLO model: {cfg.IRCAM_YOLO_PRETRAINED_WEIGHTS}")
    model = YOLO(cfg.IRCAM_YOLO_PRETRAINED_WEIGHTS)

    print(f"Starting training for IRCam model: {cfg.IRCAM_YOLO_MODEL_NAME}")
    results = model.train(
        data=cfg.IRCAM_DATA_YAML_PATH,
        epochs=cfg.IRCAM_YOLO_EPOCHS,
        imgsz=cfg.IRCAM_YOLO_IMG_SIZE,
        batch=cfg.IRCAM_YOLO_BATCH_SIZE,
        name=cfg.IRCAM_YOLO_MODEL_NAME, # Tên thử nghiệm, sẽ tạo thư mục runs/train/ir_cam_yolo_model
        project=os.path.join(cfg.PROJECT_ROOT, 'runs'), # Thư mục gốc cho các lần chạy YOLO
        exist_ok=True, # Cho phép ghi đè nếu thử nghiệm đã tồn tại
        device=0, # Sử dụng GPU 0, hoặc "cpu"
        verbose=True,
        lr0=0.001, # Các siêu tham số từ code gốc của bạn
        lrf=0.01,
        warmup_epochs=3,
        patience=20,
        augment=True,
        cos_lr=True,
        weight_decay=0.0005,
        mixup=0.1,
        mosaic=0.9, # Giảm bớt mosaic nếu ảnh IR ít đặc trưng
        copy_paste=0.1,
        cache=True # hoặc 'ram' nếu bạn có nhiều RAM
    )
    print("--- IRCam Training Finished ---")

    # YOLO tự động lưu best.pt vào runs/train/<name>/weights/best.pt
    # Sao chép mô hình tốt nhất đến thư mục models/ của dự án
    # Đường dẫn mà YOLO lưu: os.path.join(cfg.PROJECT_ROOT, 'runs', 'train', cfg.IRCAM_YOLO_MODEL_NAME, 'weights', 'best.pt')
    yolo_best_model_path = model.trainer.best # Hoặc xây dựng đường dẫn thủ công
    if os.path.exists(yolo_best_model_path):
        print(f"Copying best model from {yolo_best_model_path} to {cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH}")
        os.makedirs(os.path.dirname(cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH), exist_ok=True)
        shutil.copy2(yolo_best_model_path, cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH)
        print(f"IRCam model saved to: {cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH}")
    else:
        print(f"Lỗi: Không tìm thấy best.pt tại {yolo_best_model_path}")

    # Đánh giá mô hình trên tập test (YOLO cũng có thể làm điều này trong quá trình train)
    # print("\n--- Evaluating IRCam Model on Test Set ---")
    # test_results = model.val(data=cfg.IRCAM_DATA_YAML_PATH, split="test")
    # print(test_results.box.map50) # Ví dụ in mAP50

    print("===== IRCAM YOLO MODEL TRAINING PIPELINE FINISHED =====")

if __name__ == '__main__':
    run_ircam_yolo_training()