# Drone_Detection_Project/src/main_finetune_vcam_youtube.py
import sys
import os

PROJECT_ROOT_DIR_MAIN_FT_VCAM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR_MAIN_FT_VCAM not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR_MAIN_FT_VCAM)

try:
    from src.training.train_vcam_youtube_finetune import run_vcam_youtube_finetuning
    from src.config_loader.loader import get_config # Để tải cfg nếu cần
except ImportError as e:
    print(f"Lỗi khi import: {e}")
    exit()

if __name__ == '__main__':
    # cfg = get_config() # Tải config nếu các hàm bên trong pipeline không tự tải
    # if hasattr(cfg, 'ensure_output_directories'):
    #     cfg.ensure_output_directories()
    run_vcam_youtube_finetuning()