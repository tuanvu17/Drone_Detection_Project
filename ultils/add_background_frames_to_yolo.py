# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/add_background_frames_to_yolo.py
import os
import cv2
import glob
import shutil
import random
from tqdm import tqdm

# Import config
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_add_bg = os.path.dirname(os.path.abspath(__file__))
    src_dir_add_bg = os.path.dirname(current_dir_add_bg)
    project_root_add_bg = os.path.dirname(src_dir_add_bg)
    if project_root_add_bg not in sys.path: sys.path.insert(0, project_root_add_bg)
    from config_loader.loader import get_config

cfg_add_bg = get_config()

def count_existing_annotated_frames(dataset_base_path, splits=['train', 'val', 'test']):
    """Đếm tổng số frame hiện có CÓ FILE LABEL .txt trong các tập train, val, test."""
    total_annotated_frames = 0
    for split in splits:
        labels_dir = os.path.join(dataset_base_path, split, 'labels')
        if os.path.isdir(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
            total_annotated_frames += len(label_files)
    return total_annotated_frames

def extract_and_distribute_background_frames(
    background_video_path,
    yolo_dataset_base_path,
    target_background_percentage=0.30,
    frames_per_second_to_extract=1,
    output_image_format='.jpg'
):
    """
    Trích xuất frame từ video background, phân bổ vào các tập train, val, test
    và TẠO FILE LABEL .TXT RỖNG tương ứng.
    """
    if not os.path.exists(background_video_path):
        print(f"LỖI: Không tìm thấy video background: {background_video_path}")
        return
    if not os.path.isdir(yolo_dataset_base_path):
        print(f"LỖI: Không tìm thấy thư mục dataset YOLO: {yolo_dataset_base_path}")
        return

    print("--- Bắt đầu thêm frame BACKGROUND (và file label rỗng) vào bộ dữ liệu YOLO ---")
    print(f"Video background nguồn: {background_video_path}")
    print(f"Dataset YOLO đích: {yolo_dataset_base_path}")

    current_annotated_frames = count_existing_annotated_frames(yolo_dataset_base_path)
    if current_annotated_frames == 0:
        print("CẢNH BÁO: Không có frame nào có annotation (file .txt) trong dataset YOLO hiện tại.")
        print("  Không thể tính toán chính xác tỷ lệ background mong muốn dựa trên frame có đối tượng.")
        # Quyết định: Nếu không có frame có đối tượng, có thể không thêm frame background,
        # hoặc thêm một số lượng cố định. Hiện tại, sẽ dừng nếu không có frame có đối tượng
        # để tính tỷ lệ một cách có ý nghĩa.
        print("  Vui lòng đảm bảo dữ liệu DRONE/HELICOPTER đã được annotate và có file label.")
        return

    print(f"Số frame có annotation (đối tượng) hiện có: {current_annotated_frames}")

    if 1 - target_background_percentage <= 1e-6:
        print("LỖI: target_background_percentage quá gần hoặc bằng 1.0, không hợp lệ.")
        return
    
    num_background_frames_needed = int(round(
        (target_background_percentage * current_annotated_frames) / (1 - target_background_percentage)
    ))
    print(f"Số frame BACKGROUND cần thêm để đạt ~{target_background_percentage*100:.1f}% tổng số (so với frame có annotated): {num_background_frames_needed}")

    if num_background_frames_needed <= 0:
        print("Không cần thêm frame background hoặc số lượng tính ra không hợp lệ.")
        return

    cap = cv2.VideoCapture(background_video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video background {background_video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or video_fps is None: video_fps = 30
    frame_skip_interval = int(round(video_fps / frames_per_second_to_extract))
    if frame_skip_interval <= 0: frame_skip_interval = int(video_fps) or 1

    print(f"Video background FPS: {video_fps:.2f}. Trích xuất mỗi {frame_skip_interval} frame.")

    extracted_bg_frames_paths = []
    temp_bg_frames_dir = os.path.join(yolo_dataset_base_path, "temp_background_frames_for_script_v2")
    os.makedirs(temp_bg_frames_dir, exist_ok=True)

    frame_id_counter = 0
    saved_count = 0
    pbar_extract = tqdm(total=num_background_frames_needed, desc="Extracting BG frames", unit="frame")
    while cap.isOpened() and saved_count < num_background_frames_needed:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id_counter % frame_skip_interval == 0:
            frame_filename_base = f"background_video_{os.path.splitext(os.path.basename(background_video_path))[0]}_frame_{frame_id_counter:07d}"
            image_save_filename = f"{frame_filename_base}{output_image_format}"
            image_save_path = os.path.join(temp_bg_frames_dir, image_save_filename)
            
            # Không tạo label file ở đây, sẽ tạo khi copy vào thư mục cuối
            try:
                cv2.imwrite(image_save_path, frame)
                extracted_bg_frames_paths.append(image_save_path) # Lưu đường dẫn ảnh
                saved_count += 1
                pbar_extract.update(1)
            except Exception as e:
                print(f"Lỗi khi lưu frame background {image_save_path}: {e}")
        frame_id_counter += 1
    pbar_extract.close()
    cap.release()

    if not extracted_bg_frames_paths:
        print("Không trích xuất được frame background nào.")
        if os.path.exists(temp_bg_frames_dir): shutil.rmtree(temp_bg_frames_dir)
        return
    
    print(f"Đã trích xuất thành công {len(extracted_bg_frames_paths)} frame background vào thư mục tạm.")

    splits_info = {}
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(yolo_dataset_base_path, split, 'images')
        label_dir = os.path.join(yolo_dataset_base_path, split, 'labels')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        splits_info[split] = {'image_dir': image_dir, 'label_dir': label_dir}
    
    num_extracted_bg = len(extracted_bg_frames_paths)
    num_bg_train = int(num_extracted_bg * 0.70)
    num_bg_val = int(num_extracted_bg * 0.15)
    num_bg_test = num_extracted_bg - num_bg_train - num_bg_val
    
    if num_bg_test < 0:
        num_bg_test = 0
        if num_bg_train + num_bg_val > num_extracted_bg:
            num_bg_val = num_extracted_bg - num_bg_train
            if num_bg_val < 0:
                num_bg_train = num_extracted_bg
                num_bg_val = 0

    print(f"Phân bổ frame background dự kiến: Train={num_bg_train}, Val={num_bg_val}, Test={num_bg_test}")
    random.shuffle(extracted_bg_frames_paths)

    def copy_and_create_empty_label(source_image_paths, dest_image_dir, dest_label_dir, num_to_copy):
        copied_count = 0
        for i in range(min(num_to_copy, len(source_image_paths))):
            if not source_image_paths: break
            src_image_path = source_image_paths.pop(0)
            
            base_filename = os.path.basename(src_image_path)
            filename_no_ext = os.path.splitext(base_filename)[0]
            
            dest_image_path = os.path.join(dest_image_dir, base_filename)
            dest_label_path = os.path.join(dest_label_dir, f"{filename_no_ext}.txt") # File label rỗng
            
            try:
                shutil.copy2(src_image_path, dest_image_path)
                # Tạo file label .txt rỗng
                with open(dest_label_path, 'w') as f_label_empty:
                    pass # Chỉ cần tạo file rỗng
                copied_count +=1
            except Exception as e:
                print(f"Lỗi khi copy {src_image_path} hoặc tạo label rỗng: {e}")
        return copied_count

    print("\nCopying background frames and creating empty label files...")
    copied_train = copy_and_create_empty_label(extracted_bg_frames_paths, splits_info['train']['image_dir'], splits_info['train']['label_dir'], num_bg_train)
    print(f"  Đã copy {copied_train} frame background vào train/images và tạo label rỗng tương ứng.")
    copied_val = copy_and_create_empty_label(extracted_bg_frames_paths, splits_info['val']['image_dir'], splits_info['val']['label_dir'], num_bg_val)
    print(f"  Đã copy {copied_val} frame background vào val/images và tạo label rỗng tương ứng.")
    copied_test = copy_and_create_empty_label(extracted_bg_frames_paths, splits_info['test']['image_dir'], splits_info['test']['label_dir'], num_bg_test)
    print(f"  Đã copy {copied_test} frame background vào test/images và tạo label rỗng tương ứng.")
    
    if extracted_bg_frames_paths:
        print(f"  Cảnh báo: Còn lại {len(extracted_bg_frames_paths)} frame background chưa được phân bổ do làm tròn.")

    try:
        shutil.rmtree(temp_bg_frames_dir)
        print(f"Đã xóa thư mục tạm: {temp_bg_frames_dir}")
    except Exception as e:
        print(f"Lỗi khi xóa thư mục tạm: {e}")
    
    print("--- Thêm frame BACKGROUND và file label rỗng hoàn tất ---")
    print("\nLƯU Ý QUAN TRỌNG:")
    print("Script này đã thêm các frame ảnh BACKGROUND vào các thư mục images VÀ TẠO FILE LABEL .TXT RỖNG cho chúng.")
    print("File data_vcam_youtube.yaml không cần thay đổi 'names' hay 'nc' nếu bạn không định nghĩa BACKGROUND là một lớp có bounding box.")
    print("YOLO sẽ coi các ảnh có file label rỗng là ảnh không có đối tượng ground truth.")

if __name__ == '__main__':
    if hasattr(cfg_add_bg, 'ensure_output_directories') and callable(cfg_add_bg.ensure_output_directories):
        cfg_add_bg.ensure_output_directories()

    bg_video_path = os.path.join(cfg_add_bg.YOUTUBE_RAW_VIDEOS_BASE_DIR, 'BACKGROUND', 'Background_1.mp4')
    yolo_data_root_for_bg = cfg_add_bg.VCAM_YOUTUBE_FINETUNE_DATA_ROOT

    extract_and_distribute_background_frames(
        background_video_path=bg_video_path,
        yolo_dataset_base_path=yolo_data_root_for_bg,
        target_background_percentage=0.30,
        frames_per_second_to_extract=1
    )