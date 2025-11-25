# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/analyze_audio_dataset.py
import os
import glob
from collections import Counter

# Import config (giả sử bạn sẽ thêm đường dẫn AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR vào config)
try:
    from src.config_loader.loader import get_config
    cfg_analyze_audio = get_config()
    # Lấy đường dẫn thư mục dữ liệu audio fine-tune từ config
    # Đảm bảo bạn đã định nghĩa biến này trong project_config.py
    AUDIO_FT_SEGMENTS_DIR = getattr(cfg_analyze_audio, 'AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR', None)
    # Lấy danh sách các lớp dự kiến cho audio fine-tune từ config
    # (ví dụ: MASTER_CLASS_LIST_FUSION hoặc một biến riêng cho audio fine-tune)
    EXPECTED_AUDIO_CLASSES = getattr(cfg_analyze_audio, 'AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE',
                                     getattr(cfg_analyze_audio, 'MASTER_CLASS_LIST_FUSION', None))
except ImportError:
    print("Cảnh báo: Không thể import config, vui lòng kiểm tra sys.path hoặc cấu trúc dự án.")
    AUDIO_FT_SEGMENTS_DIR = None # Hoặc đặt đường dẫn mặc định ở đây nếu cần
    EXPECTED_AUDIO_CLASSES = None
except AttributeError as e:
    print(f"Lỗi thuộc tính trong config: {e}. Vui lòng kiểm tra file project_config.py.")
    AUDIO_FT_SEGMENTS_DIR = None
    EXPECTED_AUDIO_CLASSES = None


def analyze_audio_segments_dataset(dataset_base_path, expected_classes=None):
    """
    Phân tích bộ dữ liệu các đoạn audio đã được trích xuất và tổ chức theo lớp.

    Args:
        dataset_base_path (str): Đường dẫn đến thư mục gốc chứa các thư mục con theo lớp
                                 (ví dụ: Audio_YouTube_FineTune_Segments).
        expected_classes (list, optional): Danh sách tên các lớp dự kiến. Nếu None,
                                           script sẽ tự phát hiện các thư mục con làm tên lớp.
    Returns:
        tuple: (tổng số đoạn audio, Counter chứa số lượng đoạn mỗi lớp)
               hoặc (0, Counter()) nếu có lỗi.
    """
    if not dataset_base_path or not os.path.isdir(dataset_base_path):
        print(f"LỖI: Đường dẫn bộ dữ liệu không hợp lệ hoặc không tồn tại: {dataset_base_path}")
        return 0, Counter()

    print(f"--- Phân tích bộ dữ liệu Audio Segments tại: {dataset_base_path} ---")

    total_segments = 0
    class_counts = Counter()
    found_classes = []

    # Xác định các lớp từ thư mục con nếu expected_classes không được cung cấp
    if expected_classes is None:
        print("Đang tự động phát hiện các lớp từ thư mục con...")
        sub_items = [item for item in os.listdir(dataset_base_path)
                     if os.path.isdir(os.path.join(dataset_base_path, item))]
        if not sub_items:
            print("Không tìm thấy thư mục con nào (lớp) trong thư mục dữ liệu.")
            return 0, Counter()
        classes_to_scan = sub_items
        print(f"  Các lớp được phát hiện: {classes_to_scan}")
    else:
        classes_to_scan = expected_classes
        print(f"  Các lớp dự kiến để quét: {classes_to_scan}")


    for class_name in classes_to_scan:
        class_path = os.path.join(dataset_base_path, class_name)
        if not os.path.isdir(class_path):
            if expected_classes: # Chỉ cảnh báo nếu lớp này nằm trong danh sách dự kiến
                print(f"Cảnh báo: Thư mục cho lớp dự kiến '{class_name}' không tồn tại tại: {class_path}")
            continue

        # Đếm số lượng file .wav (hoặc các định dạng audio khác bạn dùng cho segment)
        audio_segment_files = glob.glob(os.path.join(class_path, '*.wav')) + \
                              glob.glob(os.path.join(class_path, '*.mp3')) # Thêm các định dạng khác nếu cần

        num_segments_in_class = len(audio_segment_files)
        print(f"  Lớp '{class_name}': Tìm thấy {num_segments_in_class} đoạn audio (.wav, .mp3).")

        if num_segments_in_class > 0:
            class_counts[class_name] = num_segments_in_class
            total_segments += num_segments_in_class
            if class_name not in found_classes:
                found_classes.append(class_name)

    if not found_classes and expected_classes:
        print("Cảnh báo: Không tìm thấy dữ liệu cho bất kỳ lớp dự kiến nào.")
    elif not found_classes and not expected_classes:
        print("Không tìm thấy dữ liệu trong thư mục được cung cấp.")


    return total_segments, class_counts

def main_analyze_audio_youtube_data():
    """
    Hàm chính để phân tích bộ dữ liệu Audio YouTube FineTune Segments.
    """
    if AUDIO_FT_SEGMENTS_DIR is None:
        print("LỖI: Đường dẫn AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR chưa được định nghĩa trong config hoặc không thể tải config.")
        # Cung cấp một đường dẫn mặc định nếu bạn muốn chạy script này độc lập mà không cần file config hoàn chỉnh
        # AUDIO_FT_SEGMENTS_DIR = '/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/processed/Audio_YouTube_FineTune_Segments'
        # print(f"Sử dụng đường dẫn mặc định: {AUDIO_FT_SEGMENTS_DIR}")
        # if not os.path.isdir(AUDIO_FT_SEGMENTS_DIR):
        #     print("Đường dẫn mặc định cũng không tồn tại. Dừng script.")
        #     return
        return # Dừng nếu không có đường dẫn

    # Sử dụng EXPECTED_AUDIO_CLASSES từ config nếu có, nếu không script sẽ tự phát hiện
    total_audio_segments, class_distribution = analyze_audio_segments_dataset(AUDIO_FT_SEGMENTS_DIR, EXPECTED_AUDIO_CLASSES)

    print("\n--- TỔNG KẾT BỘ DỮ LIỆU AUDIO_YOUTUBE_FINETUNE_SEGMENTS ---")
    if total_audio_segments > 0:
        print(f"Tổng số đoạn audio (1 giây) trên tất cả các lớp: {total_audio_segments}")
        if class_distribution:
            print("Phân bố đoạn audio theo lớp:")
            for class_name, count in sorted(class_distribution.items()):
                print(f"  - {class_name}: {count} đoạn")
        else:
            print("Không có thông tin phân bố lớp.")
    else:
        print("Không tìm thấy đoạn audio nào trong bộ dữ liệu.")

if __name__ == '__main__':
    main_analyze_audio_youtube_data()