# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/inspect_model_summary.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import config để lấy đường dẫn model
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    # Giả sử file này nằm trong utils/, src/ nằm cùng cấp với utils/
    current_dir_inspect = os.path.dirname(os.path.abspath(__file__))
    src_dir_inspect = os.path.dirname(current_dir_inspect) # Đi lên thư mục src/
    project_root_inspect = os.path.dirname(src_dir_inspect) # Đi lên thư mục gốc dự án
    if project_root_inspect not in sys.path: sys.path.insert(0, project_root_inspect)
    if src_dir_inspect not in sys.path: sys.path.insert(0, src_dir_inspect)
    from config_loader.loader import get_config

def print_model_summary_and_last_layers(model_path):
    """
    Tải một mô hình Keras từ đường dẫn và in ra summary của nó,
    đồng thời gợi ý tên của lớp ngay trước lớp output cuối cùng.
    """
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy file mô hình tại: {model_path}")
        return

    print(f"--- Đang tải mô hình từ: {model_path} ---")
    try:
        model = load_model(model_path)
        print("Đã tải mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    print("\n--- Model Summary ---")
    model.summary(line_length=120) # Tăng line_length để dễ đọc hơn

    # Tìm lớp ngay trước lớp output cuối cùng
    # Giả sử lớp output cuối cùng là lớp Dense duy nhất ở cuối, hoặc lớp có tên chứa "output" hoặc "softmax"/"sigmoid"
    # Đây là một phỏng đoán, bạn cần kiểm tra kỹ summary
    last_layer = model.layers[-1]
    second_to_last_layer = None

    if len(model.layers) > 1:
        second_to_last_layer = model.layers[-2] # Lớp kế cuối

    print("\n--- Gợi ý Tên Lớp Nội bộ Cuối cùng (last_internal_layer_name) ---")
    print(f"Lớp Output cuối cùng: Tên='{last_layer.name}', Loại='{type(last_layer).__name__}'")

    if second_to_last_layer:
        print(f"Lớp ngay trước lớp Output: Tên='{second_to_last_layer.name}', Loại='{type(second_to_last_layer).__name__}'")
        print(f"  => Gợi ý cho 'last_internal_layer_name_original_audio': '{second_to_last_layer.name}'")
    else:
        print("Mô hình chỉ có một lớp, không có 'lớp nội bộ cuối cùng' theo cách hiểu thông thường.")

    print("\nLƯU Ý: Hãy kiểm tra kỹ Model Summary ở trên để xác nhận tên lớp chính xác.")
    print("Bạn cần tìm tên của lớp mà output của nó sẽ là đặc trưng đầu vào cho lớp Dense output mới khi fine-tuning.")

if __name__ == '__main__':
    cfg_inspect = get_config()

    # Đường dẫn đến mô hình audio gốc của bạn
    audio_model_original_path = cfg_inspect.AUDIO_MODEL_SAVE_PATH # Lấy từ config

    print(f"Kiểm tra mô hình: {audio_model_original_path}")
    print_model_summary_and_last_layers(audio_model_original_path)

    # Nếu bạn cũng muốn kiểm tra mô hình audio đã fine-tune (ví dụ)
    # audio_model_finetuned_path = cfg_inspect.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH
    # if os.path.exists(audio_model_finetuned_path):
    #     print(f"\nKiểm tra mô hình audio đã fine-tune: {audio_model_finetuned_path}")
    #     print_model_summary_and_last_layers(audio_model_finetuned_path)
    # else:
    #     print(f"\nKhông tìm thấy mô hình audio đã fine-tune để kiểm tra tại: {audio_model_finetuned_path}")