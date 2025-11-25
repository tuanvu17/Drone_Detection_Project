# Drone_Detection_Project/src/main_finetune_fusion.py
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path để Python tìm thấy các module
# Giả sử main_finetune_fusion.py nằm trong src/
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# Import hàm pipeline chính và module tải cấu hình
try:
    from src.training.fine_tune_fusion_pipeline import run_fusion_finetuning_pipeline
    from src.config_loader.loader import get_config
except ImportError as e:
    print(f"Lỗi khi import các module cần thiết trong main_finetune_fusion.py: {e}")
    print("Hãy đảm bảo bạn đang chạy script từ thư mục gốc của dự án (Drone_Detection_Project)")
    print(f"bằng lệnh 'python -m src.main_finetune_fusion'")
    print(f"Hoặc kiểm tra sys.path hiện tại: {sys.path}")
    exit()

def main():
    print("===== BẮT ĐẦU QUY TRÌNH FINE-TUNING MÔ HÌNH EARLY FUSION (VCAM + AUDIO) =====")

    # Tải cấu hình
    try:
        cfg = get_config() # Hàm này có thể gọi ensure_output_directories nếu được thiết lập trong loader hoặc config
        print("Đã tải cấu hình dự án thành công.")
        # Gọi ensure_output_directories một cách tường minh nếu chưa được gọi khi import config
        if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories):
            print("Đảm bảo các thư mục output tồn tại...")
            cfg.ensure_output_directories()
        else:
            print("Cảnh báo: Hàm ensure_output_directories không tìm thấy hoặc không thể gọi từ config.")

    except Exception as e:
        print(f"LỖI: Không thể tải cấu hình dự án: {e}")
        print("Dừng quy trình fine-tuning.")
        return

    # Kiểm tra các file đầu vào quan trọng cho pipeline fine-tuning
    # Các file này phải được tạo bởi các pipeline trước hoặc script prepare_fusion_finetune_data.py
    # và prepare_audio_youtube_data.py (cho các thành phần audio đã fine-tune)
    required_input_files = [
        cfg.AUDIO_MODEL_FOR_FUSION_BRANCH_INIT, # Model audio đã fine-tune trên YouTube để khởi tạo nhánh
        cfg.AUDIO_SCALER_FOR_FUSION_PREP,       # Scaler audio từ quá trình fine-tune audio trên YouTube
        cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP,        # Max_len audio từ quá trình fine-tune audio trên YouTube
        cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION, # Model VCam (đã fine-tune trên YouTube) để trích xuất feature
        cfg.FINETUNE_MULTICLASS_METADATA_FILE # Metadata của dữ liệu fusion (được tạo bởi prepare_fusion_finetune_data.py)
    ]
    print("\nKiểm tra các file đầu vào cần thiết cho pipeline fine-tuning fusion:")
    all_files_exist = True
    for f_path in required_input_files:
        if not os.path.exists(f_path):
            print(f"  LỖI: Thiếu file đầu vào: {f_path}")
            all_files_exist = False
        else:
            print(f"  OK: {f_path}")

    if not all_files_exist:
        print("\nThiếu một hoặc nhiều file đầu vào quan trọng.")
        print("Vui lòng chạy các bước chuẩn bị dữ liệu và huấn luyện/fine-tune các mô hình riêng lẻ trước:")
        print(f"  - Chạy pipeline audio gốc (nếu AUDIO_MODEL_FOR_FUSION_BRANCH_INIT trỏ đến nó).")
        print(f"  - Chạy pipeline fine-tune audio trên YouTube (nếu AUDIO_MODEL_FOR_FUSION_BRANCH_INIT trỏ đến output của nó).")
        print(f"  - Chạy pipeline fine-tune VCam trên YouTube (nếu VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION trỏ đến output của nó).")
        print(f"  - Chạy prepare_fusion_finetune_data.py (để tạo {cfg.FINETUNE_MULTICLASS_METADATA_FILE} và các features).")
        print("Dừng quy trình fine-tuning fusion.")
        return

    # Gọi hàm pipeline fine-tuning fusion
    try:
        # Hàm run_fusion_finetuning_pipeline sẽ sử dụng các biến cfg đã được import trong module đó
        # Nó sẽ tự lấy các đường dẫn và tham số từ cfg.
        run_fusion_finetuning_pipeline()
        print("\n===== QUY TRÌNH FINE-TUNING MÔ HÌNH EARLY FUSION HOÀN TẤT =====")
        print(f"Mô hình fusion tốt nhất được lưu tại: {cfg.BEST_FUSION_MODEL_SAVE_PATH}")
        print(f"LabelEncoder cho fusion được lưu tại: {cfg.FUSION_LABEL_ENCODER_PATH}")
        print("Các biểu đồ và báo cáo được lưu trong thư mục reports/ tương ứng.")
    except Exception as e:
        print(f"\nLỖI Xảy ra trong quá trình fine-tuning fusion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()