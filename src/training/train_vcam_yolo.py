# Drone_Detection_Project/src/training/train_vcam_yolo.py
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

def run_vcam_yolo_training():
    print("===== STARTING VCAM YOLO MODEL TRAINING =====")
    # Gọi hàm ensure_output_directories từ module cfg
    if hasattr(cfg, 'ensure_output_directories'):
        print("Ensuring output directories from train_ircam_yolo.py...")
        cfg.ensure_output_directories()
    else:
        print("Warning: ensure_output_directories function not found in config.")

    if not os.path.exists(cfg.VCAM_DATA_YAML_PATH):
        print(f"Lỗi: Không tìm thấy file data.yaml cho VCam tại: {cfg.VCAM_DATA_YAML_PATH}")
        # ... (in hướng dẫn tạo data.yaml tương tự như IRCam) ...
        return

    print(f"Loading pretrained YOLO model: {cfg.VCAM_YOLO_PRETRAINED_WEIGHTS}")
    model = YOLO(cfg.VCAM_YOLO_PRETRAINED_WEIGHTS)

    print(f"Starting training for VCam model: {cfg.VCAM_YOLO_MODEL_NAME}")
    results = model.train(
        data=cfg.VCAM_DATA_YAML_PATH,
        epochs=cfg.VCAM_YOLO_EPOCHS,
        imgsz=cfg.VCAM_YOLO_IMG_SIZE,
        batch=cfg.VCAM_YOLO_BATCH_SIZE,
        name=cfg.VCAM_YOLO_MODEL_NAME,
        project=os.path.join(cfg.PROJECT_ROOT, 'runs'),
        exist_ok=True,
        device=0,
        verbose=True,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,
        augment=True,
        cos_lr=True,
        weight_decay=0.0005,
        mixup=0.15, # Giá trị từ code gốc của bạn
        mosaic=1.0, # Giá trị từ code gốc của bạn
        copy_paste=0.1,
        cache=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    print("--- VCam Training Finished ---")

    yolo_best_model_path = model.trainer.best
    if os.path.exists(yolo_best_model_path):
        print(f"Copying best model from {yolo_best_model_path} to {cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
        os.makedirs(os.path.dirname(cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH), exist_ok=True)
        shutil.copy2(yolo_best_model_path, cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH)
        print(f"VCam model saved to: {cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
    else:
        print(f"Lỗi: Không tìm thấy best.pt tại {yolo_best_model_path}")

    print("===== VCAM YOLO MODEL TRAINING PIPELINE FINISHED =====")

if __name__ == '__main__':
    run_vcam_yolo_training()