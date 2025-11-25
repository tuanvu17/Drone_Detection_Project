# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/analyze_yolo_dataset.py
import os
import glob
import yaml
from collections import Counter

def analyze_yolo_dataset_split(dataset_base_path, split_name, class_names_from_yaml, count_unlabelled_as_background=True):
    """
    Phân tích một tập con (train, val, test) của bộ dữ liệu YOLO.
    Đếm cả các ảnh không có file label và coi chúng là 'BACKGROUND' nếu được yêu cầu.
    """
    images_dir = os.path.join(dataset_base_path, split_name, 'images')
    labels_dir = os.path.join(dataset_base_path, split_name, 'labels')

    if not os.path.isdir(images_dir):
        print(f"Lỗi: Không tìm thấy thư mục ảnh: {images_dir}")
        return 0, Counter()

    image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(images_dir, '*.png')) + \
                  glob.glob(os.path.join(images_dir, '*.jpeg'))

    total_images_in_split = len(image_files)
    class_counts_in_split = Counter()
    unlabelled_image_count = 0

    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0: # Kiểm tra file label tồn tại và không rỗng
            try:
                with open(label_path, 'r') as f:
                    # Đếm số lượng instance mỗi lớp nếu file label có nhiều dòng
                    # Hoặc chỉ đếm lớp của đối tượng đầu tiên nếu mỗi ảnh chỉ có 1 đối tượng chính
                    # Hiện tại, ta sẽ đếm số lượng ảnh có chứa ít nhất một đối tượng của lớp đó,
                    # dựa trên dòng đầu tiên của file label.
                    lines = f.readlines()
                    processed_classes_in_this_image = set() # Để tránh đếm 1 ảnh nhiều lần cho cùng 1 lớp nếu có nhiều instance cùng lớp
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                class_id = int(line.split()[0])
                                if 0 <= class_id < len(class_names_from_yaml):
                                    class_name = class_names_from_yaml[class_id]
                                    # Để có "phân bố frame theo lớp" đúng nghĩa là mỗi frame thuộc 1 lớp chính,
                                    # logic ở đây cần phức tạp hơn (ví dụ: chỉ lấy lớp đầu tiên, hoặc lớp có diện tích lớn nhất).
                                    # Hiện tại, nếu một ảnh có cả DRONE và HELI, nó sẽ tăng count cho cả hai.
                                    # Nếu bạn muốn mỗi ảnh chỉ được gán cho 1 lớp chính trong thống kê này,
                                    # bạn cần quy tắc ưu tiên hoặc chỉ đọc dòng đầu tiên.
                                    # Giả sử chỉ đọc dòng đầu tiên cho mục đích thống kê phân bố ảnh theo lớp chính.
                                    if not processed_classes_in_this_image: # Chỉ xử lý dòng đầu tiên cho thống kê này
                                        class_counts_in_split[class_name] += 1
                                        processed_classes_in_this_image.add(class_name)
                                    break # Chỉ lấy lớp từ dòng đầu tiên cho thống kê này
                                else:
                                    print(f"Cảnh báo: Class ID {class_id} trong file {label_filename} không hợp lệ (ngoài phạm vi {len(class_names_from_yaml)} lớp).")
                            except ValueError:
                                print(f"Cảnh báo: Dòng không hợp lệ trong file label {label_filename}: '{line}'")
            except Exception as e:
                print(f"Lỗi khi đọc file nhãn {label_filename}: {e}")
        else: # Ảnh không có file label hoặc file label rỗng
            unlabelled_image_count += 1

    if count_unlabelled_as_background and unlabelled_image_count > 0:
        class_counts_in_split['BACKGROUND'] = unlabelled_image_count
        print(f"  Tìm thấy {unlabelled_image_count} frame không có label (được tính là BACKGROUND).")

    return total_images_in_split, class_counts_in_split

def main_analyze_vcam_youtube_data():
    dataset_base_path = cfg_add_bg.VCAM_YOUTUBE_FINETUNE_DATA_ROOT # Sử dụng biến config từ add_background script
    yaml_path = cfg_add_bg.VCAM_YOUTUBE_DATA_YAML_PATH

    if not os.path.exists(yaml_path):
        print(f"LỖI: Không tìm thấy file data_vcam_youtube.yaml tại: {yaml_path}")
        return
    try:
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        class_names_from_yaml = data_yaml.get('names', [])
        if not class_names_from_yaml:
            print("LỖI: Không tìm thấy 'names' (danh sách lớp) trong file data.yaml.")
            return
        print(f"Các lớp được định nghĩa trong data.yaml (dùng để ánh xạ ID): {class_names_from_yaml}")
    except Exception as e:
        print(f"Lỗi khi đọc file data.yaml: {e}")
        return

    print(f"\n--- Phân tích bộ dữ liệu: {dataset_base_path} ---")

    total_images_all_splits_overall = 0
    class_counts_all_splits_overall = Counter()

    for split in ['train', 'val', 'test']:
        print(f"\n--- Phân tích tập: {split.upper()} ---")
        # Mặc định count_unlabelled_as_background=True để thống kê cả ảnh nền
        total_images_in_current_split, class_counts_in_current_split = \
            analyze_yolo_dataset_split(dataset_base_path, split, class_names_from_yaml, count_unlabelled_as_background=True)

        if total_images_in_current_split > 0:
            print(f"Tổng số frame (ảnh) trong tập {split.upper()}: {total_images_in_current_split}")
            if class_counts_in_current_split:
                print("Phân bố frame theo lớp (bao gồm cả BACKGROUND nếu có):")
                for class_name, count in sorted(class_counts_in_current_split.items()):
                    print(f"  - {class_name}: {count} frame")
            else:
                print("  Không có thông tin phân bố lớp chi tiết.")
            total_images_all_splits_overall += total_images_in_current_split
            class_counts_all_splits_overall.update(class_counts_in_current_split)
        else:
            print(f"Không tìm thấy ảnh nào trong tập {split.upper()}.")

    print("\n--- TỔNG KẾT TOÀN BỘ DATASET VCam_YouTube_FineTune_Data ---")
    print(f"Tổng số frame (ảnh) trên tất cả các tập: {total_images_all_splits_overall}")
    if class_counts_all_splits_overall:
        print("Tổng phân bố frame theo lớp trên tất cả các tập (bao gồm BACKGROUND nếu có):")
        for class_name, count in sorted(class_counts_all_splits_overall.items()):
            print(f"  - {class_name}: {count} frame")
    else:
        print("Không có thông tin phân bố lớp tổng thể.")
from src.config_loader.loader import get_config
if __name__ == '__main__':
    cfg_add_bg = get_config() # Tải config để lấy đường dẫn
    main_analyze_vcam_youtube_data()