# Drone_Detection_Project/config/project_config.py
import os

# --- Đường dẫn Gốc ---
# Lấy đường dẫn thư mục gốc của dự án (giả sử config.py nằm trong config/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Cấu hình Chung ---
CLASSES_ALL = ['DRONE', 'HELICOPTER', 'BACKGROUND'] # Các lớp chung, có thể dùng cho nhiều modal

# === Cấu hình cho Dữ liệu và Mô hình Âm thanh ===
AUDIO_SUBDIR_NAME = 'Audio_Original' # Tên thư mục con chứa dữ liệu audio thô
AUDIO_RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', AUDIO_SUBDIR_NAME)
AUDIO_CLASSES = CLASSES_ALL # Sử dụng các lớp chung
AUDIO_SAMPLE_RATE = 44100
AUDIO_SEGMENT_DURATION = 1.0
AUDIO_N_MFCC = 13
AUDIO_N_FFT = int(AUDIO_SAMPLE_RATE * 0.025)
AUDIO_HOP_LENGTH = int(AUDIO_SAMPLE_RATE * 0.010)

AUDIO_VALIDATION_SPLIT = 0.20
AUDIO_NUM_TEST_FILES_PER_CLASS = 5

AUDIO_EPOCHS = 50
AUDIO_BATCH_SIZE = 32
AUDIO_LEARNING_RATE = 0.001

AUDIO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'audio_classifier')
AUDIO_MODEL_SAVE_PATH = os.path.join(AUDIO_MODEL_DIR, 'best_audio_model.keras')
AUDIO_LABEL_ENCODER_PATH = os.path.join(AUDIO_MODEL_DIR, 'label_encoder_audio.pkl')
AUDIO_SCALER_PATH = os.path.join(AUDIO_MODEL_DIR, 'scaler_audio.pkl')
AUDIO_MAX_LEN_PATH = os.path.join(AUDIO_MODEL_DIR, 'max_len_audio.pkl')

AUDIO_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
AUDIO_TEST_FILES_LIST_PATH = os.path.join(AUDIO_PROCESSED_DATA_DIR, 'test_files_list_audio.txt')

AUDIO_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
AUDIO_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics')

AUDIO_TRAINING_HISTORY_PLOT_PATH = os.path.join(AUDIO_REPORTS_FIGURES_DIR, 'audio_training_history.png')
AUDIO_CONFUSION_MATRIX_VAL_PLOT_PATH = os.path.join(AUDIO_REPORTS_FIGURES_DIR, 'audio_confusion_matrix_val.png')
AUDIO_CONFUSION_MATRIX_TEST_PLOT_PATH = os.path.join(AUDIO_REPORTS_FIGURES_DIR, 'audio_confusion_matrix_test.png')
AUDIO_DETAILED_TEST_RESULTS_PATH = os.path.join(AUDIO_REPORTS_METRICS_DIR, 'audio_detailed_test_results.txt')

AUDIO_PREDICTIONS_SAVE_PATH = os.path.join(AUDIO_MODEL_DIR, 'audio_test_predictions.pkl')
AUDIO_TRUE_LABELS_SAVE_PATH = os.path.join(AUDIO_MODEL_DIR, 'audio_test_true_labels.pkl')
AUDIO_CLASS_NAMES_SAVE_PATH = os.path.join(AUDIO_MODEL_DIR, 'audio_test_class_names.pkl')

# === Cấu hình cho Dữ liệu và Mô hình VCam (Sẽ thêm sau) ===
# VCAM_RAW_DATA_PATH = ...
# VCAM_MODEL_SAVE_PATH = ...

# === Cấu hình cho Dữ liệu và Mô hình IRCam (Sẽ thêm sau) ===
# IRCAM_RAW_DATA_PATH = ...
# IRCAM_MODEL_SAVE_PATH = ...


# # --- Đảm bảo các thư mục lưu trữ tồn tại (có thể gọi khi cần) ---
# def ensure_output_directories():
#     dirs_to_create = [
#         AUDIO_MODEL_DIR, AUDIO_PROCESSED_DATA_DIR,
#         AUDIO_REPORTS_FIGURES_DIR, AUDIO_REPORTS_METRICS_DIR,
#         # Thêm các thư mục cho VCam, IRCam sau này
#     ]
#     for d in dirs_to_create:
#         os.makedirs(d, exist_ok=True)
#     print("Output directories checked/created.")

# # Gọi hàm này một lần khi script config được import lần đầu (hoặc gọi trong script chính)
# # ensure_output_directories() # Hoặc bạn có thể gọi nó ở đầu mỗi script training 

# --- Cấu hình Chung ---
CLASSES_YOLO = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
NUM_CLASSES_YOLO = len(CLASSES_YOLO)

# === Cấu hình cho Dữ liệu và Mô hình Âm thanh (Giữ nguyên) ===
# ... (phần audio giữ nguyên) ...
AUDIO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'audio_classifier') # Ví dụ, cần cho ensure_output_directories
AUDIO_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed') # Ví dụ
AUDIO_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures') # Ví dụ
AUDIO_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics') # Ví dụ


# === Cấu hình cho Dữ liệu và Mô hình IRCam (YOLO) ===
IRCAM_PROCESSED_DATA_SUBDIR_NAME = 'ir_cam' # Đặt tên thư mục con
IRCAM_PROCESSED_DATA_FULL_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'VCam_IRCam_Processed', IRCAM_PROCESSED_DATA_SUBDIR_NAME)
IRCAM_DATA_YAML_PATH = os.path.join(IRCAM_PROCESSED_DATA_FULL_PATH, 'data.yaml')
IRCAM_VIDEOS_TO_PREDICT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'IR_Videos_Original', 'VIDEO2PREDICT')

IRCAM_YOLO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'ircam_yolo_classifier')
IRCAM_YOLO_MODEL_NAME = 'ir_cam_yolo_model'
IRCAM_YOLO_BEST_MODEL_SAVE_PATH = os.path.join(IRCAM_YOLO_MODEL_DIR, f'{IRCAM_YOLO_MODEL_NAME}_best.pt')

IRCAM_YOLO_IMG_SIZE = 416
IRCAM_YOLO_BATCH_SIZE = 16
IRCAM_YOLO_EPOCHS = 100
IRCAM_YOLO_PRETRAINED_WEIGHTS = "yolo11n.pt" 

IRCAM_YOLO_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'ircam_yolo')
IRCAM_YOLO_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'ircam_yolo')
IRCAM_YOLO_RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs', 'train') # Thêm đường dẫn cho YOLO runs

# === Cấu hình cho Dữ liệu và Mô hình VCam (YOLO) ===
VCAM_PROCESSED_DATA_SUBDIR_NAME = 'v_cam' # Đặt tên thư mục con
VCAM_PROCESSED_DATA_FULL_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'VCam_IRCam_Processed', VCAM_PROCESSED_DATA_SUBDIR_NAME)
VCAM_DATA_YAML_PATH = os.path.join(VCAM_PROCESSED_DATA_FULL_PATH, 'data.yaml')
VCAM_VIDEOS_TO_PREDICT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'V_Videos_Original', 'VIDEO2PREDICT')

VCAM_YOLO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'vcam_yolo_classifier')
VCAM_YOLO_MODEL_NAME = 'v_cam_yolo_model'
VCAM_YOLO_BEST_MODEL_SAVE_PATH = os.path.join(VCAM_YOLO_MODEL_DIR, f'{VCAM_YOLO_MODEL_NAME}_best.pt')

VCAM_YOLO_IMG_SIZE = 640
VCAM_YOLO_BATCH_SIZE = 16
VCAM_YOLO_EPOCHS = 100
VCAM_YOLO_PRETRAINED_WEIGHTS = "yolo11n.pt"

VCAM_YOLO_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'vcam_yolo')
VCAM_YOLO_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'vcam_yolo')
VCAM_YOLO_RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs', 'train') # Thêm đường dẫn cho YOLO runs


# --- Đảm bảo các thư mục lưu trữ tồn tại ---
def ensure_output_directories():
    # Truy cập trực tiếp các biến đã định nghĩa ở trên trong cùng module này
    dirs_to_create = [
        AUDIO_MODEL_DIR, AUDIO_PROCESSED_DATA_DIR, # Ví dụ từ phần audio
        AUDIO_REPORTS_FIGURES_DIR, AUDIO_REPORTS_METRICS_DIR,

        IRCAM_YOLO_MODEL_DIR, VCAM_YOLO_MODEL_DIR,
        IRCAM_PROCESSED_DATA_FULL_PATH, # Thư mục chứa data.yaml và train/val/test
        VCAM_PROCESSED_DATA_FULL_PATH,
        IRCAM_VIDEOS_TO_PREDICT_DIR, VCAM_VIDEOS_TO_PREDICT_DIR,
        IRCAM_YOLO_REPORTS_FIGURES_DIR, IRCAM_YOLO_REPORTS_METRICS_DIR,
        VCAM_YOLO_REPORTS_FIGURES_DIR, VCAM_YOLO_REPORTS_METRICS_DIR,
        IRCAM_YOLO_RUNS_DIR, VCAM_YOLO_RUNS_DIR # Thư mục runs/train mà YOLO sẽ sử dụng
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    print("Output directories checked/created from project_config.py.")

# Bạn có thể chọn gọi hàm này ở cuối file project_config.py
# để nó tự động chạy khi module config được import lần đầu tiên,
# hoặc gọi nó một cách tường minh ở đầu các script main.
# Ví dụ:
# if __name__ != "__main__": # Chỉ chạy khi được import, không chạy khi thực thi trực tiếp file này
#     ensure_output_directories()