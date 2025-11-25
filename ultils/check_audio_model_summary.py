# Drone_Detection_Project/check_audio_model_summary.py
import os
import sys

# Thêm thư mục gốc của dự án vào sys.path để Python tìm thấy module 'config'
# Điều này quan trọng nếu bạn chạy script này từ thư_mục_gốc/
# hoặc nếu 'config' không nằm trong PYTHONPATH mặc định.
PROJECT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__)) # Đường dẫn đến thư mục chứa file này
# Nếu file này nằm trong src/, thì PROJECT_ROOT_DIR sẽ là src/
# Nếu file này nằm trong Drone_Detection_Project/, thì PROJECT_ROOT_DIR là Drone_Detection_Project/
# Giả sử bạn đặt file này trong Drone_Detection_Project/ để dễ chạy ban đầu
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR) # Thêm thư mục gốc dự án

from tensorflow.keras.models import load_model
from config import project_config as cfg # Đảm bảo config.project_config có thể được import

print(f"Đang cố gắng tải mô hình từ: {cfg.AUDIO_MODEL_SAVE_PATH}")

try:
    # Đảm bảo thư mục models và các thư mục con đã được tạo bởi ensure_output_directories
    if hasattr(cfg, 'ensure_output_directories'):
        print("Kiểm tra và tạo các thư mục output nếu cần...")
        cfg.ensure_output_directories()

    if not os.path.exists(cfg.AUDIO_MODEL_SAVE_PATH):
        print(f"LỖI: File mô hình không tồn tại tại đường dẫn: {cfg.AUDIO_MODEL_SAVE_PATH}")
        print("Vui lòng kiểm tra lại đường dẫn trong config/project_config.py hoặc đảm bảo mô hình đã được huấn luyện và lưu.")
    else:
        audio_model = load_model(cfg.AUDIO_MODEL_SAVE_PATH)
        print("\n--- Tóm tắt Kiến trúc Mô hình Âm thanh (best_audio_model.keras) ---")
        audio_model.summary(line_length=120)
        print("\n--- Kết thúc Tóm tắt ---")
        print("Hãy tìm tên của lớp BiLSTM cuối cùng (trước các lớp Dense phân loại) từ summary ở trên.")
        print("Ví dụ, nó có thể là 'bidirectional_2' hoặc một tên tương tự.")

except ImportError as e_imp:
    print(f"Lỗi ImportError: Không thể import 'config.project_config'. Lỗi: {e_imp}")
    print(f"sys.path hiện tại: {sys.path}")
    print("Hãy đảm bảo file config/project_config.py tồn tại và thư mục 'config' có file __init__.py.")
    print("Đồng thời, đảm bảo thư mục gốc của dự án (Drone_Detection_Project) nằm trong PYTHONPATH hoặc bạn đang chạy script từ thư mục đó.")
except FileNotFoundError as e_fnf:
    print(f"Lỗi FileNotFoundError: {e_fnf}")
    print(f"Vui lòng kiểm tra lại đường dẫn '{cfg.AUDIO_MODEL_SAVE_PATH}' trong config/project_config.py.")
except Exception as e:
    print(f"Lỗi không xác định khi tải hoặc xem summary model audio: {e}")