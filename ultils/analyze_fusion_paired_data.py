# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/analyze_fusion_paired_data.py
import os
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split # Để mô phỏng cách chia nếu cần
from sklearn.preprocessing import LabelEncoder # Để làm việc với nhãn nếu cần
import numpy as np
# Import config
try:
    from src.config_loader.loader import get_config
    cfg_analyze_fusion = get_config()
    # Lấy các đường dẫn và tham số từ config
    FINETUNE_METADATA_FILE = getattr(cfg_analyze_fusion, 'FINETUNE_MULTICLASS_METADATA_FILE', None)
    # Tỷ lệ chia cho Fusion (để ước tính Train/Val/Test nếu metadata không chứa thông tin này)
    # Giả sử bạn có các biến này trong config, nếu không, đặt giá trị mặc định
    FUSION_TEST_SPLIT = getattr(cfg_analyze_fusion, 'FUSION_TEST_IN_DOMAIN_SPLIT_RATIO', 0.20) # Ví dụ 20% cho Test
    FUSION_VAL_ON_REMAINING_SPLIT = getattr(cfg_analyze_fusion, 'FUSION_VALIDATION_ON_REMAINING_RATIO', 0.20) # Ví dụ 20% của phần còn lại cho Val
    FUSION_CLASSES = getattr(cfg_analyze_fusion, 'FUSION_CLASS_NAMES', None)

except ImportError:
    print("Cảnh báo: Không thể import config. Sử dụng đường dẫn mặc định.")
    FINETUNE_METADATA_FILE = '/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/processed/FineTune_Paired_Data/finetune_multiclass_metadata.csv'
    FUSION_TEST_SPLIT = 0.20
    FUSION_VAL_ON_REMAINING_SPLIT = 0.20
    FUSION_CLASSES = None # Sẽ cố gắng lấy từ dữ liệu
except AttributeError as e:
    print(f"Lỗi thuộc tính trong config: {e}. Vui lòng kiểm tra file project_config.py.")
    FINETUNE_METADATA_FILE = None


def analyze_paired_fusion_dataset(metadata_file_path,
                                  test_split_ratio,
                                  validation_on_remaining_ratio,
                                  expected_classes=None):
    """
    Phân tích file metadata của bộ dữ liệu ghép cặp cho Fusion.
    Ước tính số lượng mẫu cho Train, Validation, và Test In-Domain.
    """
    if not metadata_file_path or not os.path.exists(metadata_file_path):
        print(f"LỖI: File metadata không tồn tại: {metadata_file_path}")
        return

    print(f"--- Phân tích Bộ Dữ liệu Ghép cặp Ảnh-Âm thanh cho Fusion ---")
    print(f"Từ file metadata: {metadata_file_path}")

    try:
        df = pd.read_csv(metadata_file_path)
        print(f"Đã đọc {len(df)} dòng (segment) từ metadata.")
    except Exception as e:
        print(f"Lỗi khi đọc file CSV metadata: {e}")
        return

    if 'label' not in df.columns:
        print("LỖI: Cột 'label' không tồn tại trong file metadata.")
        return

    total_segments = len(df)
    print(f"Tổng số cặp (segment) ảnh-âm thanh: {total_segments}")

    class_counts_total = Counter(df['label'])
    print("\nPhân bố tổng thể theo lớp:")
    if class_counts_total:
        for class_name, count in sorted(class_counts_total.items()):
            print(f"  - {class_name}: {count} segment")
    else:
        print("  Không có thông tin phân bố lớp.")

    # Ước tính số lượng cho Train, Validation, Test In-Domain
    # Lưu ý: Đây là ước tính dựa trên tỷ lệ. Script huấn luyện thực tế
    # có thể có số lượng hơi khác do làm tròn hoặc cách stratify.
    if total_segments == 0:
        print("Không có dữ liệu để phân chia.")
        return

    labels_for_splitting = df['label'].tolist()
    indices = np.arange(total_segments)

    # Mã hóa nhãn để stratify (nếu cần)
    le = LabelEncoder()
    if expected_classes:
        le.fit(expected_classes) # Ưu tiên fit trên danh sách lớp đầy đủ từ config
        try:
            y_encoded_for_split = le.transform(labels_for_splitting)
        except ValueError:
            print("Cảnh báo: Có nhãn trong metadata không nằm trong expected_classes. Stratify có thể không chính xác.")
            y_encoded_for_split = le.fit_transform(labels_for_splitting) # Fit lại trên dữ liệu thực tế
    else:
        y_encoded_for_split = le.fit_transform(labels_for_splitting)


    num_test_samples = 0
    num_val_samples = 0
    num_train_samples = 0

    # Phân chia Test In-Domain
    if test_split_ratio > 0 and total_segments * test_split_ratio >= 1 :
        try:
            # Chỉ lấy indices để ước tính, không cần dữ liệu thực
            train_val_indices, test_indices, \
            y_train_val_temp, y_test_temp = train_test_split(
                indices, y_encoded_for_split,
                test_size=test_split_ratio,
                random_state=42, # Giữ random_state nhất quán với pipeline huấn luyện
                stratify=y_encoded_for_split
            )
            num_test_samples = len(test_indices)
            labels_in_test = y_encoded_for_split[test_indices]
            class_counts_test = Counter(le.inverse_transform(labels_in_test))

            # Phần còn lại cho Train và Validation
            remaining_labels_for_train_val = y_train_val_temp # Đây là y_encoded_for_split[train_val_indices]
            remaining_indices_for_train_val = train_val_indices

        except ValueError as e_split_test:
            print(f"Cảnh báo: Lỗi khi ước tính chia Test ({e_split_test}). Giả sử không có tập Test riêng từ bước này.")
            remaining_labels_for_train_val = y_encoded_for_split
            remaining_indices_for_train_val = indices
            class_counts_test = Counter() # Test rỗng
    else:
        print("Không tách tập Test In-Domain (do tỷ lệ bằng 0 hoặc không đủ dữ liệu).")
        remaining_labels_for_train_val = y_encoded_for_split
        remaining_indices_for_train_val = indices
        class_counts_test = Counter()

    # Phân chia Train và Validation từ phần còn lại
    if validation_on_remaining_ratio > 0 and len(remaining_labels_for_train_val) * validation_on_remaining_ratio >=1 :
        try:
            _, _, \
            y_train_temp, y_val_temp = train_test_split(
                remaining_indices_for_train_val, # Chỉ cần indices để lấy y
                remaining_labels_for_train_val,
                test_size=validation_on_remaining_ratio,
                random_state=42,
                stratify=remaining_labels_for_train_val
            )
            num_train_samples = len(y_train_temp)
            num_val_samples = len(y_val_temp)
            class_counts_train = Counter(le.inverse_transform(y_train_temp))
            class_counts_val = Counter(le.inverse_transform(y_val_temp))

        except ValueError as e_split_val:
            print(f"Cảnh báo: Lỗi khi ước tính chia Validation ({e_split_val}). Toàn bộ phần còn lại là Train.")
            num_train_samples = len(remaining_labels_for_train_val)
            class_counts_train = Counter(le.inverse_transform(remaining_labels_for_train_val))
            class_counts_val = Counter() # Val rỗng
    else:
        print("Không tách tập Validation (do tỷ lệ bằng 0 hoặc không đủ dữ liệu còn lại).")
        num_train_samples = len(remaining_labels_for_train_val)
        class_counts_train = Counter(le.inverse_transform(remaining_labels_for_train_val))
        class_counts_val = Counter()


    print("\n--- Ước tính Quy mô và Phân bố cho các Tập (Fusion In-Domain) ---")
    print(f"Tổng số cặp (segment) ảnh-âm thanh dự kiến cho Huấn luyện (Train_Fusion_YT): {num_train_samples}")
    if class_counts_train:
        for cls, count in sorted(class_counts_train.items()): print(f"  - {cls}: {count} segment")

    print(f"\nTổng số cặp (segment) ảnh-âm thanh dự kiến cho Kiểm chứng (Val_Fusion_YT): {num_val_samples}")
    if class_counts_val:
        for cls, count in sorted(class_counts_val.items()): print(f"  - {cls}: {count} segment")

    print(f"\nTổng số cặp (segment) ảnh-âm thanh dự kiến cho Kiểm thử (Test_Fusion_YT - In-Domain): {num_test_samples}")
    if class_counts_test:
        for cls, count in sorted(class_counts_test.items()): print(f"  - {cls}: {count} segment")
    
    print("\nLƯU Ý: Số liệu trên là ước tính dựa trên tỷ lệ chia. Số lượng thực tế trong quá trình huấn luyện có thể hơi khác một chút do cách hàm train_test_split xử lý các trường hợp biên hoặc làm tròn khi stratify với số lượng mẫu nhỏ cho từng lớp.")
    print("Hãy sử dụng số liệu thực tế được in ra bởi pipeline huấn luyện fusion để báo cáo chính xác nhất.")

if __name__ == '__main__':
    if FINETUNE_METADATA_FILE is None:
        print("LỖI: Đường dẫn FINETUNE_MULTICLASS_METADATA_FILE chưa được định nghĩa trong config hoặc không thể tải config.")
    else:
        analyze_paired_fusion_dataset(
            metadata_file_path=FINETUNE_METADATA_FILE,
            test_split_ratio=FUSION_TEST_SPLIT,
            validation_on_remaining_ratio=FUSION_VAL_ON_REMAINING_SPLIT,
            expected_classes=FUSION_CLASSES
        )