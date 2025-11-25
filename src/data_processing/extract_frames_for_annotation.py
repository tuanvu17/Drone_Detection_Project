# Drone_Detection_Project/src/data_processing/extract_frames_for_annotation.py
import os
import glob
import cv2
from tqdm import tqdm

# Import config để lấy đường dẫn
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    # ... (xử lý sys.path nếu cần) ...
    from config_loader.loader import get_config

cfg = get_config()

# --- Cấu hình cho script này ---
# Thư mục chứa các video raw từ YouTube, được tổ chức theo lớp
SOURCE_VIDEO_BASE_DIR = cfg.YOUTUBE_RAW_VIDEOS_BASE_DIR # Từ project_config.py

# Thư mục output để lưu các frame đã trích xuất (bạn sẽ gán nhãn trong thư mục này)
# Tạo một thư mục riêng để tránh nhầm lẫn với dữ liệu đã xử lý cho YOLO
FRAMES_FOR_ANNOTATION_DIR = os.path.join(cfg.PROJECT_ROOT, 'data', 'interim', 'Frames_For_VCam_Annotation')

# Các lớp bạn muốn trích xuất frame (lấy từ config hoặc định nghĩa ở đây)
# Giả sử bạn muốn xử lý tất cả các lớp có video trong YOUTUBE_RAW_VIDEOS_BASE_DIR
# Hoặc bạn có thể chỉ định cụ thể: CLASSES_TO_EXTRACT = ['DRONE', 'HELICOPTER']
CLASSES_TO_EXTRACT = cfg.MASTER_CLASS_LIST_FUSION # Hoặc cfg.CLASSES_YOLO

FRAME_SAVE_INTERVAL = 180 # Lưu 1 frame mỗi N frame (ví dụ: nếu video 30fps, 15 ~ 2 frame/giây)
                           # Điều chỉnh cho phù hợp để không có quá nhiều frame giống nhau
                           # Hoặc bạn có thể đặt là 1 để lấy tất cả các frame nếu video ngắn

def extract_frames():
    print(f"===== BẮT ĐẦU TRÍCH XUẤT FRAME ĐỂ ANNOTATION =====")
    print(f"Thư mục video nguồn: {SOURCE_VIDEO_BASE_DIR}")
    print(f"Thư mục lưu frame output: {FRAMES_FOR_ANNOTATION_DIR}")

    if not os.path.exists(SOURCE_VIDEO_BASE_DIR):
        print(f"LỖI: Thư mục video nguồn không tồn tại: {SOURCE_VIDEO_BASE_DIR}")
        return

    os.makedirs(FRAMES_FOR_ANNOTATION_DIR, exist_ok=True)
    total_frames_extracted = 0

    for class_name in CLASSES_TO_EXTRACT:
        class_video_dir = os.path.join(SOURCE_VIDEO_BASE_DIR, class_name)
        output_class_frame_dir = os.path.join(FRAMES_FOR_ANNOTATION_DIR, class_name) # Lưu frame theo lớp
        os.makedirs(output_class_frame_dir, exist_ok=True)

        if not os.path.isdir(class_video_dir):
            print(f"  Cảnh báo: Không tìm thấy thư mục video cho lớp '{class_name}' tại '{class_video_dir}'. Bỏ qua.")
            continue

        video_files = glob.glob(os.path.join(class_video_dir, "*.*"))
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print(f"  Không tìm thấy video nào cho lớp '{class_name}'.")
            continue

        print(f"\n--- Đang xử lý lớp: {class_name} ---")
        for video_path in video_files:
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"  Trích xuất frame từ video: {video_base_name}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    Lỗi: Không thể mở video {video_base_name}. Bỏ qua.")
                continue

            frame_idx = 0
            saved_frame_count_for_video = 0
            pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"  {video_base_name}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % FRAME_SAVE_INTERVAL == 0:
                    frame_filename = f"{video_base_name}_frame_{frame_idx:06d}.jpg"
                    save_path = os.path.join(output_class_frame_dir, frame_filename)
                    cv2.imwrite(save_path, frame)
                    saved_frame_count_for_video += 1
                
                frame_idx += 1
                pbar.update(1)
            
            pbar.close()
            cap.release()
            print(f"    Đã trích xuất {saved_frame_count_for_video} frame từ {video_base_name}.")
            total_frames_extracted += saved_frame_count_for_video
            
    print(f"\n===== TRÍCH XUẤT FRAME HOÀN TẤT =====")
    print(f"Tổng cộng {total_frames_extracted} frame đã được lưu vào: {FRAMES_FOR_ANNOTATION_DIR}")
    print("Bây giờ bạn có thể sử dụng một công cụ annotation (như LabelImg, CVAT) để gán nhãn bounding box cho các frame trong thư mục này.")

if __name__ == '__main__':
    # Đảm bảo các thư mục output chính đã được tạo bởi config
    if hasattr(cfg, 'ensure_output_directories'):
        cfg.ensure_output_directories()
    extract_frames()