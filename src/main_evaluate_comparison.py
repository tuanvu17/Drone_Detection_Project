# Drone_Detection_Project/src/main_evaluate_comparison.py
import os
import sys

# Đặt os.environ TRƯỚC khi import TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Buộc TensorFlow chạy trên CPU

# Thêm thư mục gốc của dự án vào sys.path
PROJECT_ROOT_DIR_MAIN_EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR_MAIN_EVAL not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR_MAIN_EVAL)

try:
    from src.evaluation.compare_models_on_out_of_domain import run_model_comparison_on_evaluation_set
    from src.config_loader.loader import get_config # Để tải cfg nếu hàm con không tự tải
except ImportError as e:
    print(f"Lỗi khi import các module cần thiết: {e}")
    print("Hãy đảm bảo bạn đang chạy script từ thư mục gốc của dự án (Drone_Detection_Project)")
    print(f"bằng lệnh 'python -m src.main_evaluate_comparison'")
    exit()

def main():
    print("===== BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH =====")
    
    # Tải cấu hình (để đảm bảo các thư mục output được tạo nếu cần)
    try:
        cfg_main_eval = get_config()
        if hasattr(cfg_main_eval, 'ensure_output_directories') and callable(cfg_main_eval.ensure_output_directories):
            print("Đảm bảo các thư mục output tồn tại...")
            cfg_main_eval.ensure_output_directories()
    except Exception as e_cfg:
        print(f"LỖI: Không thể tải hoặc xử lý cấu hình: {e_cfg}")
        return

    # Kiểm tra các file đầu vào chính cần thiết cho toàn bộ quá trình so sánh
    required_overall_files = [
        cfg_main_eval.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE,
        cfg_main_eval.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH, # Detector VCam
        cfg_main_eval.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION,   # VCam feature extractor cho Fusion
        cfg_main_eval.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH,  # Model Audio fine-tuned
        cfg_main_eval.AUDIO_YOUTUBE_SCALER_PATH,
        cfg_main_eval.AUDIO_YOUTUBE_MAX_LEN_PATH,
        cfg_main_eval.BEST_FUSION_MODEL_SAVE_PATH,              # Model Fusion
        cfg_main_eval.FUSION_LABEL_ENCODER_PATH
    ]
    print("\nKiểm tra các file đầu vào cần thiết cho pipeline so sánh model:")
    all_main_files_exist = True
    for f_path_main in required_overall_files:
        if not os.path.exists(f_path_main):
            print(f"  LỖI: Thiếu file đầu vào: {f_path_main}")
            all_main_files_exist = False
        else:
            print(f"  OK: {f_path_main}")

    if not all_main_files_exist:
        print("\nThiếu một hoặc nhiều file đầu vào quan trọng. Vui lòng chạy các bước chuẩn bị và huấn luyện trước.")
        print("Dừng quy trình đánh giá so sánh.")
        return
    
    try:
        # Chạy trên CPU cho các model PyTorch
        run_model_comparison_on_evaluation_set(device_for_pytorch_models='cpu')
        print("\n===== QUY TRÌNH ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH HOÀN TẤT =====")
    except Exception as e_compare:
        print(f"\nLỖI Xảy ra trong quá trình so sánh mô hình: {e_compare}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()