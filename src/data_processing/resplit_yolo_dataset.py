# src/data_processing/resplit_yolo_dataset.py
import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import random

def resplit_dataset(source_base_dir, target_base_dir, test_ratio, val_on_remaining_ratio, random_state=42):
    print(f"Resplitting dataset from: {source_base_dir}")
    print(f"Target directory: {target_base_dir}")

    source_images_dir = os.path.join(source_base_dir, 'images')
    source_labels_dir = os.path.join(source_base_dir, 'labels')

    if not os.path.isdir(source_images_dir) or not os.path.isdir(source_labels_dir):
        print("Lỗi: Thư mục images hoặc labels nguồn không tồn tại.")
        return

    all_image_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(source_images_dir, '*.jpg')) + \
                              glob.glob(os.path.join(source_images_dir, '*.png'))]) # Thêm các định dạng ảnh khác nếu có
    if not all_image_files:
        print("Không tìm thấy file ảnh nào trong thư mục nguồn.")
        return
    
    # Tạo nhãn giả để stratify (vì YOLO không có nhãn file tổng thể, chúng ta làm trên từng file)
    # Nếu muốn stratify theo lớp, bạn cần đọc từng file label để xác định lớp đa số trong ảnh, phức tạp hơn.
    # Cách đơn giản là chia ngẫu nhiên danh sách file ảnh.
    # Để đơn giản, chúng ta sẽ không stratify ở bước này, mà dựa vào sự ngẫu nhiên.
    # Nếu bạn có thông tin lớp cho từng ảnh, bạn có thể tạo y_labels ở đây để stratify.
    
    # Bước 1: Tách Test
    remaining_files, test_files = train_test_split(
        all_image_files,
        test_size=test_ratio,
        random_state=random_state
    )

    # Bước 2: Tách Train và Validation từ phần còn lại
    # Tỷ lệ val cho lần chia thứ 2 phải được tính lại trên tập remaining_files
    # Ví dụ: nếu test_ratio = 0.3, remaining là 0.7
    # val_on_remaining_ratio = 0.2 (20% của 70%)
    # => test_size cho lần chia thứ 2 = 0.2
    # Hoặc val_ratio_on_original = 0.14 (14% của gốc)
    # new_val_ratio = val_ratio_on_original / (1 - test_ratio)
    # Vì val_on_remaining_ratio đã là tỷ lệ trên phần còn lại, ta dùng trực tiếp
    
    if not remaining_files: # Nếu không còn file nào sau khi tách test
        print("Không còn file nào cho train/validation sau khi tách test.")
        train_files = []
        val_files = []
    elif len(remaining_files) < 2: # Không đủ để chia train/val
        print("Chỉ còn 1 file sau khi tách test, dùng làm train.")
        train_files = remaining_files
        val_files = []
    else:
        try:
            train_files, val_files = train_test_split(
                remaining_files,
                test_size=val_on_remaining_ratio, # 20% của phần còn lại
                random_state=random_state
            )
        except ValueError: # Xảy ra nếu val_on_remaining_ratio làm cho 1 tập rỗng
            print(f"Cảnh báo: Không thể chia validation với val_on_remaining_ratio={val_on_remaining_ratio}. Dùng tất cả còn lại cho train.")
            train_files = remaining_files
            val_files = []


    print(f"Tổng số file ban đầu: {len(all_image_files)}")
    print(f"  Train: {len(train_files)} files")
    print(f"  Validation: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")

    # Tạo thư mục đích và copy file
    for subset_name, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        target_img_subset_dir = os.path.join(target_base_dir, subset_name, 'images')
        target_lbl_subset_dir = os.path.join(target_base_dir, subset_name, 'labels')
        os.makedirs(target_img_subset_dir, exist_ok=True)
        os.makedirs(target_lbl_subset_dir, exist_ok=True)

        for img_basename in file_list:
            img_name_no_ext = Path(img_basename).stem
            lbl_basename = img_name_no_ext + ".txt"

            src_img = os.path.join(source_images_dir, img_basename)
            src_lbl = os.path.join(source_labels_dir, lbl_basename)

            if os.path.exists(src_img) and os.path.exists(src_lbl):
                shutil.copy2(src_img, os.path.join(target_img_subset_dir, img_basename))
                shutil.copy2(src_lbl, os.path.join(target_lbl_subset_dir, lbl_basename))
            else:
                print(f"Cảnh báo: Thiếu file ảnh hoặc nhãn cho {img_basename} / {lbl_basename}")
    print(f"Đã chia lại dataset và lưu vào: {target_base_dir}")

if __name__ == '__main__':
    try:
        from src.config_loader.loader import get_config
        cfg_resplit = get_config()
        # Thư mục chứa toàn bộ ảnh và label đã remapped ID
        SOURCE_DATA_FOR_RESPLIT = os.path.join(cfg_resplit.PROJECT_ROOT, 'data', 'interim', 'VCam_YouTube_All_Remapped')
        # Thư mục đích cho dataset đã được chia lại theo tỷ lệ mới
        TARGET_DATA_RESPLIT = cfg_resplit.VCAM_YOUTUBE_FINETUNE_DATA_FULL_PATH
        
        TEST_RATIO_CFG = 0.30
        VAL_ON_REMAINING_RATIO_CFG = 0.20 # 20% của 70% còn lại

        # Tạo thư mục nguồn giả định (bạn cần copy dữ liệu vào đây trước)
        # os.makedirs(os.path.join(SOURCE_DATA_FOR_RESPLIT, 'images'), exist_ok=True)
        # os.makedirs(os.path.join(SOURCE_DATA_FOR_RESPLIT, 'labels'), exist_ok=True)
        # print(f"Hãy copy TẤT CẢ các file ảnh và label (đã remapped ID) vào {SOURCE_DATA_FOR_RESPLIT}")
        # input("Nhấn Enter sau khi đã copy xong...")

    except:
        print("Không thể tải config, sử dụng đường dẫn mặc định cho resplitting.")
        PROJECT_ROOT_RESPLIT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        SOURCE_DATA_FOR_RESPLIT = os.path.join(PROJECT_ROOT_RESPLIT, 'data', 'interim', 'VCam_YouTube_All_Remapped')
        TARGET_DATA_RESPLIT = os.path.join(PROJECT_ROOT_RESPLIT, 'data', 'processed', 'VCam_YouTube_FineTune_Data')
        TEST_RATIO_CFG = 0.30
        VAL_ON_REMAINING_RATIO_CFG = 0.20

    if not os.path.exists(SOURCE_DATA_FOR_RESPLIT) or \
       not os.path.exists(os.path.join(SOURCE_DATA_FOR_RESPLIT, 'images')) or \
       not os.path.exists(os.path.join(SOURCE_DATA_FOR_RESPLIT, 'labels')):
        print(f"LỖI: Thư mục nguồn cho việc chia lại không hợp lệ hoặc thiếu thư mục con images/labels: {SOURCE_DATA_FOR_RESPLIT}")
        print("Vui lòng tạo thư mục này và copy toàn bộ ảnh, nhãn (đã remapped ID) vào đó.")
    else:
         # Xóa thư mục đích cũ nếu có để tránh file bị lẫn
        if os.path.exists(TARGET_DATA_RESPLIT):
            print(f"Xóa thư mục đích cũ: {TARGET_DATA_RESPLIT}")
            shutil.rmtree(TARGET_DATA_RESPLIT)
        resplit_dataset(SOURCE_DATA_FOR_RESPLIT, TARGET_DATA_RESPLIT, TEST_RATIO_CFG, VAL_ON_REMAINING_RATIO_CFG)