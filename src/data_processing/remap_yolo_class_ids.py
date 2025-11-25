# src/data_processing/remap_yolo_class_ids.py
import os
import glob

def remap_ids_in_file(file_path, id_mapping):
    lines_out = []
    changed = False
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            try:
                original_id = int(parts[0])
                if original_id in id_mapping:
                    new_id = id_mapping[original_id]
                    parts[0] = str(new_id)
                    lines_out.append(" ".join(parts))
                    changed = True
                else:
                    lines_out.append(line.strip()) # Giữ nguyên nếu ID không có trong mapping
            except ValueError:
                lines_out.append(line.strip()) # Giữ nguyên nếu dòng không hợp lệ
    
    if changed:
        with open(file_path, 'w') as f:
            for line_out in lines_out:
                f.write(line_out + '\n')
        # print(f"  Updated: {file_path}")

def run_remap_for_dataset(dataset_base_path, id_mapping):
    print(f"Remapping Class IDs in dataset: {dataset_base_path}")
    for subset in ['train', 'val', 'test']: # Bao gồm cả test nếu có
        labels_dir = os.path.join(dataset_base_path, subset, 'labels')
        if not os.path.isdir(labels_dir):
            print(f"  Warning: Labels directory not found for {subset}: {labels_dir}")
            continue
        
        print(f"  Processing subset: {subset}")
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        if not label_files:
            print(f"    No label files found in {labels_dir}")
            continue

        for label_file in label_files:
            remap_ids_in_file(label_file, id_mapping)
    print("Class ID remapping completed.")

if __name__ == '__main__':
    # Cấu hình đường dẫn và ánh xạ ID
    # Lấy từ config hoặc định nghĩa trực tiếp ở đây để chạy riêng lẻ
    try:
        from src.config_loader.loader import get_config
        cfg_remap = get_config()
        VARS_DATASET_PATH = cfg_remap.VCAM_YOUTUBE_FINETUNE_DATA_FULL_PATH # Đường dẫn đến VCam_YouTube_FineTune_Data
    except:
        print("Không thể tải config, sử dụng đường dẫn mặc định cho remapping.")
        PROJECT_ROOT_REMAP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        VARS_DATASET_PATH = os.path.join(PROJECT_ROOT_REMAP, 'data', 'processed', 'VCam_YouTube_FineTune_Data')


    # Ánh xạ từ ID của Roboflow (0: DRONE, 1: HELICOPTER)
    # sang ID mong muốn cho mô hình pre-trained 4 lớp
    # (AIRPLANE:0, BIRD:1, DRONE:2, HELICOPTER:3)
    ID_MAPPING_ROBOFLOW_TO_FULL = {
        0: 2,  # DRONE từ Roboflow (ID 0) sẽ thành ID 2
        1: 3   # HELICOPTER từ Roboflow (ID 1) sẽ thành ID 3
    }
    if os.path.exists(VARS_DATASET_PATH):
        run_remap_for_dataset(VARS_DATASET_PATH, ID_MAPPING_ROBOFLOW_TO_FULL)
    else:
        print(f"LỖI: Thư mục dataset không tồn tại: {VARS_DATASET_PATH}")