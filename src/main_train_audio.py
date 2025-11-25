# Drone_Detection_Project/src/main_train_audio.py
import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
# Giả sử main_train_audio.py nằm trong src/
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR) # Thêm thư mục gốc dự án (Drone_Detection_Project)

# Import từ các module trong src và config
from src.training.train_audio_pipeline import run_audio_training_pipeline
from src.evaluation.evaluate_audio import test_audio_model_on_saved_files # THÊM IMPORT NÀY
from config import project_config as cfg # SỬA LẠI ĐỂ IMPORT TRỰC TIẾP cfg

def main():
    print("Bắt đầu Quy trình Huấn luyện và Đánh giá Mô hình Âm thanh...")

    # Đảm bảo các thư mục output tồn tại (có thể gọi hàm từ cfg nếu có)
    if hasattr(cfg, 'ensure_output_directories'):
        cfg.ensure_output_directories()
    else: # Hoặc tạo thủ công ở đây nếu hàm đó không có/không được gọi trong cfg
        dirs_to_create = [
            cfg.AUDIO_MODEL_DIR, cfg.AUDIO_PROCESSED_DATA_DIR,
            cfg.AUDIO_REPORTS_FIGURES_DIR_ORIGINAL, cfg.AUDIO_REPORTS_METRICS_DIR_ORIGINAL,
        ]
        for d_path in dirs_to_create:
            os.makedirs(d_path, exist_ok=True)

    RUN_TRAINING = True
    RUN_TESTING = True

    if RUN_TRAINING:
        run_audio_training_pipeline(
            base_path=cfg.AUDIO_RAW_DATA_PATH,
            classes=cfg.AUDIO_CLASSES_ORIGINAL,
            sample_rate=cfg.AUDIO_SAMPLE_RATE,
            segment_duration=cfg.AUDIO_SEGMENT_DURATION,
            n_mfcc=cfg.AUDIO_N_MFCC,
            n_fft=cfg.AUDIO_N_FFT, # Truyền giá trị đã tính
            hop_length=cfg.AUDIO_HOP_LENGTH, # Truyền giá trị đã tính
            # validation_split=cfg.AUDIO_VALIDATION_SPLIT,
            validation_split=cfg.AUDIO_ORIGINAL_VALIDATION_ON_REMAINING_RATIO,
            num_test_files_per_class=cfg.AUDIO_ORIGINAL_NUM_TEST_FILES_PER_CLASS,
            epochs=cfg.AUDIO_ORIGINAL_EPOCHS,
            batch_size=cfg.AUDIO_ORIGINAL_BATCH_SIZE,
            learning_rate=cfg.AUDIO_ORIGINAL_LEARNING_RATE,
            model_save_path=cfg.AUDIO_MODEL_SAVE_PATH,
            label_encoder_path=cfg.AUDIO_LABEL_ENCODER_PATH,
            scaler_path=cfg.AUDIO_SCALER_PATH,
            max_len_path=cfg.AUDIO_MAX_LEN_PATH,
            test_files_list_path=cfg.AUDIO_ORIGINAL_TEST_FILES_LIST_PATH,
            training_history_plot_path=cfg.AUDIO_TRAINING_HISTORY_PLOT_PATH_ORIGINAL,
            confusion_matrix_val_plot_path=cfg.AUDIO_CONFUSION_MATRIX_VAL_PLOT_PATH_ORIGINAL,
            # Các tham số cho hàm test cũng được truyền vào pipeline training
            # để nó có thể gọi hàm test ở cuối nếu muốn (hoặc bỏ nếu test chạy riêng)
            confusion_matrix_test_plot_path=cfg.AUDIO_CONFUSION_MATRIX_TEST_PLOT_PATH_ORIGINAL,
            detailed_test_results_path=cfg.AUDIO_DETAILED_TEST_RESULTS_PATH_ORIGINAL,
            predictions_save_path=cfg.AUDIO_PREDICTIONS_SAVE_PATH_ORIGINAL,
            true_labels_save_path=cfg.AUDIO_TRUE_LABELS_SAVE_PATH_ORIGINAL,
            class_names_save_path=cfg.AUDIO_CLASS_NAMES_SAVE_PATH_ORIGINAL
        )

    if RUN_TESTING:
        print("\n===== STARTING AUDIO MODEL EVALUATION ON TEST SET (called from main) =====")
        required_files_for_test = [
            cfg.AUDIO_MODEL_SAVE_PATH, cfg.AUDIO_LABEL_ENCODER_PATH,
            cfg.AUDIO_SCALER_PATH, cfg.AUDIO_MAX_LEN_PATH, cfg.AUDIO_ORIGINAL_TEST_FILES_LIST_PATH
        ]
        if all(os.path.exists(f) for f in required_files_for_test):
            test_audio_model_on_saved_files( # Gọi hàm test trực tiếp từ main
                base_path=cfg.AUDIO_RAW_DATA_PATH, # Đường dẫn đến thư mục chứa các lớp audio thô
                model_path=cfg.AUDIO_MODEL_SAVE_PATH,
                label_encoder_path=cfg.AUDIO_LABEL_ENCODER_PATH,
                scaler_path=cfg.AUDIO_SCALER_PATH,
                max_len_path=cfg.AUDIO_MAX_LEN_PATH,
                test_files_list_path=cfg.AUDIO_ORIGINAL_TEST_FILES_LIST_PATH,
                segment_duration=cfg.AUDIO_SEGMENT_DURATION,
                sr=cfg.AUDIO_SAMPLE_RATE,
                n_mfcc=cfg.AUDIO_N_MFCC,
                n_fft=cfg.AUDIO_N_FFT,
                hop_length=cfg.AUDIO_HOP_LENGTH,
                results_save_path=cfg.AUDIO_DETAILED_TEST_RESULTS_PATH_ORIGINAL,
                predictions_save_path=cfg.AUDIO_PREDICTIONS_SAVE_PATH_ORIGINAL,
                true_labels_save_path=cfg.AUDIO_TRUE_LABELS_SAVE_PATH_ORIGINAL,
                class_names_save_path=cfg.AUDIO_CLASS_NAMES_SAVE_PATH_ORIGINAL,
                confusion_matrix_save_path=cfg.AUDIO_CONFUSION_MATRIX_TEST_PLOT_PATH_ORIGINAL
            )
        else:
            print("ERROR: Not all required files for testing are present. Please run training first or check paths.")
            for f_path_check in required_files_for_test:
                if not os.path.exists(f_path_check): print(f"Missing: {f_path_check}")
        print("===== AUDIO MODEL EVALUATION (from main) FINISHED =====")

    print("\n===== SCRIPT FINISHED =====")

if __name__ == "__main__":
    main()