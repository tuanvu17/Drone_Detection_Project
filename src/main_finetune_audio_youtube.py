# Drone_Detection_Project/src/main_finetune_audio_youtube.py
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
PROJECT_ROOT_DIR_MAIN_FT_AUDIO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR_MAIN_FT_AUDIO not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR_MAIN_FT_AUDIO)

try:
    from src.training.fine_tune_audio_youtube_pipeline import run_audio_youtube_finetuning_pipeline
    from src.config_loader.loader import get_config # Để tải cfg nếu các hàm con không tự tải
except ImportError as e:
    print(f"Lỗi khi import các module cần thiết: {e}")
    exit()

def main():
    print("===== BẮT ĐẦU QUY TRÌNH FINE-TUNING MÔ HÌNH AUDIO TRÊN DỮ LIỆU YOUTUBE =====")
    
    # Tải cấu hình (để đảm bảo các thư mục output được tạo nếu cần)
    try:
        cfg = get_config() # Hàm này có thể gọi ensure_output_directories
        print("Đã tải cấu hình dự án thành công.")
    except Exception as e:
        print(f"LỖI: Không thể tải cấu hình dự án: {e}")
        print("Dừng quy trình fine-tuning audio YouTube.")
        return

    # Kiểm tra file đầu vào quan trọng
    # Các file này phải được tạo bởi các pipeline trước hoặc script prepare_audio_youtube_data.py
    required_input_files = [
        cfg.AUDIO_MODEL_SAVE_PATH, # Mô hình audio gốc để bắt đầu fine-tune
        cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE # Metadata của dữ liệu audio YouTube
    ]
    print("\nKiểm tra các file đầu vào cần thiết cho pipeline fine-tuning audio YouTube:")
    all_files_exist = True
    for f_path in required_input_files:
        if not os.path.exists(f_path):
            print(f"  LỖI: Thiếu file đầu vào: {f_path}")
            all_files_exist = False
        else:
            print(f"  OK: {f_path}")

    if not all_files_exist:
        print("\nThiếu một hoặc nhiều file đầu vào quan trọng. Vui lòng chạy các bước chuẩn bị trước.")
        print("Dừng quy trình fine-tuning audio YouTube.")
        return

    try:
        run_audio_youtube_finetuning_pipeline() # Hàm này sẽ sử dụng cfg đã được import trong module đó
        print("\n===== QUY TRÌNH FINE-TUNING MÔ HÌNH AUDIO TRÊN DỮ LIỆU YOUTUBE HOÀN TẤT =====")
        print(f"Mô hình audio đã fine-tune được lưu tại: {cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"\nLỖI Xảy ra trong quá trình fine-tuning audio YouTube: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()