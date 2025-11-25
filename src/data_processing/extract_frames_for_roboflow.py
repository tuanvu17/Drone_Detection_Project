# Drone_Detection_Project/src/data_processing/extract_frames_for_roboflow.py
import os
import cv2
from tqdm import tqdm

# Import config
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_extract = os.path.dirname(os.path.abspath(__file__))
    src_dir_extract = os.path.dirname(current_dir_extract)
    project_root_extract = os.path.dirname(src_dir_extract)
    if project_root_extract not in sys.path: sys.path.insert(0, project_root_extract)
    if src_dir_extract not in sys.path: sys.path.insert(0, src_dir_extract)
    from config_loader.loader import get_config

cfg_extract = get_config()

def extract_frames(
    source_video_base_dir,
    output_frames_base_dir,
    frames_per_second_to_extract=1 # Trích xuất 1 frame mỗi giây
):
    """
    Đọc các video từ source_video_base_dir, trích xuất một số lượng frame nhất định
    mỗi giây và lưu chúng vào output_frames_base_dir với cấu trúc thư mục theo lớp.
    """
    if not os.path.exists(source_video_base_dir):
        print(f"LỖI: Thư mục video nguồn không tồn tại: {source_video_base_dir}")
        return

    print(f"Bắt đầu trích xuất frame từ: {source_video_base_dir}")
    print(f"Frame sẽ được lưu vào: {output_frames_base_dir}")
    print(f"Số frame trích xuất mỗi giây: {frames_per_second_to_extract}")

    # Duyệt qua các thư mục lớp trong source_video_base_dir
    for class_name in os.listdir(source_video_base_dir):
        class_video_dir = os.path.join(source_video_base_dir, class_name)
        if not os.path.isdir(class_video_dir):
            continue

        # Tạo thư mục output cho từng lớp
        output_class_frames_dir = os.path.join(output_frames_base_dir, class_name)
        os.makedirs(output_class_frames_dir, exist_ok=True)
        print(f"\nĐang xử lý lớp: {class_name}")

        video_files = [f for f in os.listdir(class_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            print(f"  Không tìm thấy file video nào trong thư mục lớp: {class_name}")
            continue

        for video_filename in tqdm(video_files, desc=f"Extracting frames from {class_name}"):
            video_path = os.path.join(class_video_dir, video_filename)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Cảnh báo: Không thể mở video {video_path}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps == 0: # Xử lý trường hợp không lấy được FPS
                print(f"Cảnh báo: Không thể lấy FPS từ video {video_filename}. Sử dụng frame_skip mặc định là 30.")
                frame_skip_interval = 30 # Mặc định
            else:
                frame_skip_interval = int(round(video_fps / frames_per_second_to_extract))
            
            if frame_skip_interval <= 0: # Đảm bảo frame_skip_interval > 0
                frame_skip_interval = 1
                print(f"Cảnh báo: frame_skip_interval tính được <=0 cho video {video_filename} (FPS: {video_fps}). Đặt lại thành 1 (trích xuất mọi frame).")


            print(f"  Video: {video_filename}, FPS: {video_fps:.2f}, Trích xuất mỗi {frame_skip_interval} frame.")

            frame_id_counter = 0 # Đếm frame trong video hiện tại
            saved_frame_count_for_video = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id_counter % frame_skip_interval == 0:
                    # Tạo tên file cho frame
                    frame_filename = f"{os.path.splitext(video_filename)[0]}_frame_{frame_id_counter:06d}.jpg" # Lưu dưới dạng jpg
                    save_path = os.path.join(output_class_frames_dir, frame_filename)
                    try:
                        cv2.imwrite(save_path, frame)
                        saved_frame_count_for_video += 1
                    except Exception as e:
                        print(f"Lỗi khi lưu frame {save_path}: {e}")
                
                frame_id_counter += 1
            
            cap.release()
            if saved_frame_count_for_video > 0:
                print(f"    Đã trích xuất và lưu {saved_frame_count_for_video} frame từ {video_filename}")
            else:
                print(f"    Không trích xuất được frame nào (hoặc không lưu được) từ {video_filename} với frame_skip_interval={frame_skip_interval}")


    print("\n===== TRÍCH XUẤT FRAME HOÀN TẤT =====")

if __name__ == '__main__':
    # Đảm bảo các thư mục output được tạo nếu hàm ensure_output_directories không được gọi tự động khi import config
    if hasattr(cfg_extract, 'ensure_output_directories'):
        cfg_extract.ensure_output_directories()
    else: # Tự tạo các thư mục cần thiết cho script này
        os.makedirs(cfg_extract.EVALUATION_OOD_FRAMES_FOR_ROBOFLOW_DIR, exist_ok=True)
        # Tạo thư mục con cho từng lớp nếu bạn muốn (Roboflow có thể tự nhận dạng nếu bạn upload theo cấu trúc đó)
        # Hoặc bạn có thể để Roboflow tự tạo cấu trúc lớp khi bạn upload và gán nhãn.
        # Để đơn giản cho việc upload ban đầu, ta sẽ tạo thư mục con theo lớp.
        if os.path.exists(cfg_extract.EVALUATION_OOD_RAW_VIDEO_BASE_DIR):
            for class_n in os.listdir(cfg_extract.EVALUATION_OOD_RAW_VIDEO_BASE_DIR):
                if os.path.isdir(os.path.join(cfg_extract.EVALUATION_OOD_RAW_VIDEO_BASE_DIR, class_n)):
                    os.makedirs(os.path.join(cfg_extract.EVALUATION_OOD_FRAMES_FOR_ROBOFLOW_DIR, class_n), exist_ok=True)


    extract_frames(
        source_video_base_dir=cfg_extract.EVALUATION_OOD_RAW_VIDEO_BASE_DIR,
        output_frames_base_dir=cfg_extract.EVALUATION_OOD_FRAMES_FOR_ROBOFLOW_DIR,
        frames_per_second_to_extract=1 # Trích xuất 1 frame mỗi giây
    )