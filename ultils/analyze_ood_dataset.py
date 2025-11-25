# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/analyze_ood_dataset.py
import os
import pandas as pd
from collections import Counter

# Import config
try:
    from src.config_loader.loader import get_config
    cfg_analyze_ood = get_config()
    # Lấy đường dẫn đến file metadata OOD và danh sách lớp từ config
    OOD_METADATA_FILE_PATH = getattr(cfg_analyze_ood, 'EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE', None)
    # Danh sách lớp này nên là MASTER_CLASS_LIST_FUSION để bao quát tất cả các lớp mục tiêu
    OOD_EXPECTED_CLASSES = getattr(cfg_analyze_ood, 'MASTER_CLASS_LIST_FUSION', None)
except ImportError:
    print("Cảnh báo: Không thể import config. Sử dụng đường dẫn mặc định.")
    OOD_METADATA_FILE_PATH = '/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/processed/Evaluation_Test_Segments/evaluation_test_ground_truth.csv'
    OOD_EXPECTED_CLASSES = None # Sẽ cố gắng lấy từ dữ liệu nếu config không có
except AttributeError as e:
    print(f"Lỗi thuộc tính trong config: {e}. Vui lòng kiểm tra file project_config.py.")
    OOD_METADATA_FILE_PATH = None
    OOD_EXPECTED_CLASSES = None

def analyze_ood_evaluation_dataset(metadata_file_path, expected_classes=None):
    """
    Phân tích file metadata của bộ dữ liệu Test Out-of-Domain.
    Thống kê tổng số segment và phân bố theo lớp.
    """
    if not metadata_file_path or not os.path.exists(metadata_file_path):
        print(f"LỖI: File metadata OOD không tồn tại: {metadata_file_path}")
        return

    print(f"--- Phân tích Bộ Dữ liệu Test Out-of-Domain (OOD) ---")
    print(f"Từ file metadata: {metadata_file_path}")

    try:
        df = pd.read_csv(metadata_file_path)
        print(f"Đã đọc {len(df)} dòng (segment) từ metadata OOD.")
    except Exception as e:
        print(f"Lỗi khi đọc file CSV metadata OOD: {e}")
        return

    # Giả sử cột chứa nhãn ground truth trong file CSV này là 'ground_truth_label'
    # (dựa trên tên file config EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE)
    # Hoặc bạn có thể cần điều chỉnh tên cột này nếu nó khác
    label_column_name = 'ground_truth_label'
    if label_column_name not in df.columns:
        # Thử một tên cột phổ biến khác nếu 'ground_truth_label' không có
        if 'label' in df.columns:
            label_column_name = 'label'
        else:
            print(f"LỖI: Cột nhãn ('{label_column_name}' hoặc 'label') không tồn tại trong file metadata OOD.")
            print(f"Các cột có sẵn: {df.columns.tolist()}")
            return

    total_segments = len(df)
    print(f"Tổng số segment trong bộ OOD: {total_segments}")

    if total_segments == 0:
        print("Không có dữ liệu trong file metadata OOD để phân tích.")
        return

    class_counts_ood = Counter(df[label_column_name].astype(str).str.strip()) # Đảm bảo là string và loại bỏ khoảng trắng

    print("\nPhân bố segment theo lớp trong bộ OOD:")
    if class_counts_ood:
        # Xác định các lớp sẽ hiển thị
        if expected_classes and isinstance(expected_classes, list):
            # Hiển thị theo thứ tự của expected_classes và bao gồm cả lớp có count = 0
            for class_name in expected_classes:
                count = class_counts_ood.get(class_name, 0) # Lấy count, mặc định là 0 nếu lớp không có trong dữ liệu
                print(f"  - {class_name}: {count} segment")
            # Kiểm tra xem có lớp nào trong dữ liệu mà không có trong expected_classes không
            unexpected_classes_in_data = [cls for cls in class_counts_ood if cls not in expected_classes]
            if unexpected_classes_in_data:
                print("\n  Cảnh báo: Tìm thấy các lớp trong dữ liệu OOD không có trong danh sách lớp dự kiến:")
                for cls_name in unexpected_classes_in_data:
                    print(f"    - {cls_name}: {class_counts_ood[cls_name]} segment")
        else:
            # Nếu không có expected_classes, chỉ hiển thị các lớp có trong dữ liệu
            print("  (Không có danh sách lớp dự kiến, hiển thị các lớp tìm thấy trong dữ liệu)")
            for class_name, count in sorted(class_counts_ood.items()):
                print(f"  - {class_name}: {count} segment")
    else:
        print("  Không có thông tin phân bố lớp trong dữ liệu OOD.")

    print("\n--- Phân tích OOD hoàn tất ---")

if __name__ == '__main__':
    if OOD_METADATA_FILE_PATH is None:
        print("LỖI: Đường dẫn EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE chưa được định nghĩa trong config hoặc không thể tải config.")
    else:
        analyze_ood_evaluation_dataset(
            metadata_file_path=OOD_METADATA_FILE_PATH,
            expected_classes=OOD_EXPECTED_CLASSES # Truyền danh sách lớp từ config
        )