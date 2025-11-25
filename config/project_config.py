# Drone_Detection_Project/config/project_config.py
import os

# --- Đường dẫn Gốc ---
# Lấy đường dẫn thư mục gốc của dự án (giả sử config.py nằm trong config/, nên đi lên 2 cấp)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Cấu hình Chung ---
# Danh sách TẤT CẢ các lớp bạn dự định cho mô hình FUSION cuối cùng
# Thứ tự này QUAN TRỌNG và sẽ được dùng cho LabelEncoder của Fusion model
MASTER_CLASS_LIST_FUSION = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER', 'BACKGROUND']

# Danh sách lớp cho YOLO (có thể giống hoặc khác một chút)
CLASSES_YOLO = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
NUM_CLASSES_YOLO = len(CLASSES_YOLO)


# === Cấu hình cho Dữ liệu và Mô hình Âm thanh (GỐC) ===
AUDIO_SUBDIR_NAME = 'Audio_Original' # Thư mục con chứa dữ liệu audio gốc (DRONE, HELI, BG)
AUDIO_RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', AUDIO_SUBDIR_NAME)
AUDIO_CLASSES_ORIGINAL = ['DRONE', 'HELICOPTER', 'BACKGROUND'] # Các lớp của mô hình audio gốc
AUDIO_SAMPLE_RATE = 44100
AUDIO_SEGMENT_DURATION = 1.0 # Giữ nguyên cho tất cả xử lý audio segment
AUDIO_N_MFCC = 13
AUDIO_N_FFT = int(AUDIO_SAMPLE_RATE * 0.025)
AUDIO_HOP_LENGTH = int(AUDIO_SAMPLE_RATE * 0.010)

# Cấu hình chia dữ liệu cho mô hình Audio GỐC
AUDIO_ORIGINAL_TEST_SPLIT_RATIO = 0.30
AUDIO_ORIGINAL_VALIDATION_ON_REMAINING_RATIO = 0.20

AUDIO_ORIGINAL_EPOCHS = 50 # Epochs cho mô hình audio gốc
AUDIO_ORIGINAL_BATCH_SIZE = 32
AUDIO_ORIGINAL_LEARNING_RATE = 0.001
AUDIO_ORIGINAL_NUM_TEST_FILES_PER_CLASS = 5 # Ví dụ: 5 file test cho mỗi lớp, Đảm bảo tính độc lập của Tập Test


AUDIO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'audio_classifier') # Nơi lưu model audio GỐC
AUDIO_MODEL_SAVE_PATH = os.path.join(AUDIO_MODEL_DIR, 'best_audio_model.keras') # Model audio GỐC
AUDIO_LABEL_ENCODER_PATH = os.path.join(AUDIO_MODEL_DIR, 'label_encoder_audio_original.pkl') # Encoder cho model audio GỐC
AUDIO_SCALER_PATH = os.path.join(AUDIO_MODEL_DIR, 'scaler_audio_original.pkl') # Scaler cho model audio GỐC
AUDIO_MAX_LEN_PATH = os.path.join(AUDIO_MODEL_DIR, 'max_len_audio_original.pkl') # MaxLen cho model audio GỐC

AUDIO_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed') # Thư mục chung cho processed data
# Danh sách file cho mô hình audio gốc
AUDIO_ORIGINAL_TRAIN_FILES_LIST_PATH = os.path.join(AUDIO_PROCESSED_DATA_DIR, 'audio_original_train_files.txt')
AUDIO_ORIGINAL_VAL_FILES_LIST_PATH = os.path.join(AUDIO_PROCESSED_DATA_DIR, 'audio_original_val_files.txt')
AUDIO_ORIGINAL_TEST_FILES_LIST_PATH = os.path.join(AUDIO_PROCESSED_DATA_DIR, 'audio_original_test_files.txt')

AUDIO_REPORTS_FIGURES_DIR_ORIGINAL = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'audio_original')
AUDIO_REPORTS_METRICS_DIR_ORIGINAL = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'audio_original')
AUDIO_TRAINING_HISTORY_PLOT_PATH_ORIGINAL = os.path.join(AUDIO_REPORTS_FIGURES_DIR_ORIGINAL, 'audio_original_training_history.png')
AUDIO_CONFUSION_MATRIX_VAL_PLOT_PATH_ORIGINAL = os.path.join(AUDIO_REPORTS_FIGURES_DIR_ORIGINAL, 'audio_original_confusion_matrix_val.png')
AUDIO_CONFUSION_MATRIX_TEST_PLOT_PATH_ORIGINAL = os.path.join(AUDIO_REPORTS_FIGURES_DIR_ORIGINAL, 'audio_original_confusion_matrix_test.png')
AUDIO_DETAILED_TEST_RESULTS_PATH_ORIGINAL = os.path.join(AUDIO_REPORTS_METRICS_DIR_ORIGINAL, 'audio_original_detailed_test_results.txt')
# Các file lưu kết quả predict của model audio gốc (nếu có bước evaluate riêng)
AUDIO_PREDICTIONS_SAVE_PATH_ORIGINAL = os.path.join(AUDIO_MODEL_DIR, 'audio_original_test_predictions.pkl')
AUDIO_TRUE_LABELS_SAVE_PATH_ORIGINAL = os.path.join(AUDIO_MODEL_DIR, 'audio_original_test_true_labels.pkl')
AUDIO_CLASS_NAMES_SAVE_PATH_ORIGINAL = os.path.join(AUDIO_MODEL_DIR, 'audio_original_test_class_names.pkl')


# === Cấu hình chung cho Dữ liệu Ảnh đã xử lý (YOLO format) ===
PROCESSED_IMAGE_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed', 'VCam_IRCam_Processed')

# === Cấu hình cho IRCam (YOLO) ===
IRCAM_PROCESSED_DATA_SUBDIR_NAME = 'ir_cam'
IRCAM_PROCESSED_DATA_FULL_PATH = os.path.join(PROCESSED_IMAGE_DATA_ROOT, IRCAM_PROCESSED_DATA_SUBDIR_NAME)
IRCAM_DATA_YAML_PATH = os.path.join(IRCAM_PROCESSED_DATA_FULL_PATH, 'data.yaml')
IRCAM_VIDEOS_TO_PREDICT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'IR_Videos_Original', 'VIDEO2PREDICT')
IRCAM_YOLO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'ircam_yolo_classifier')
IRCAM_YOLO_MODEL_NAME = 'ir_cam_yolo_model'
IRCAM_YOLO_BEST_MODEL_SAVE_PATH = os.path.join(IRCAM_YOLO_MODEL_DIR, f'{IRCAM_YOLO_MODEL_NAME}_best.pt')
IRCAM_YOLO_PRETRAINED_WEIGHTS = "yolo11n.pt"
IRCAM_YOLO_IMG_SIZE = 416
IRCAM_YOLO_EPOCHS = 100
IRCAM_YOLO_BATCH_SIZE = 16
IRCAM_YOLO_RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs', 'train_ircam')
IRCAM_YOLO_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'ircam_yolo')

# === Cấu hình cho VCam (YOLO - GỐC, huấn luyện trên dữ liệu ban đầu) ===
VCAM_PROCESSED_DATA_SUBDIR_NAME_ORIGINAL = 'v_cam_original_data' # Phân biệt với dữ liệu YouTube
VCAM_PROCESSED_DATA_FULL_PATH_ORIGINAL = os.path.join(PROCESSED_IMAGE_DATA_ROOT, VCAM_PROCESSED_DATA_SUBDIR_NAME_ORIGINAL)
VCAM_DATA_YAML_PATH_ORIGINAL = os.path.join(VCAM_PROCESSED_DATA_FULL_PATH_ORIGINAL, 'data.yaml')
VCAM_VIDEOS_TO_PREDICT_DIR_ORIGINAL = os.path.join(PROJECT_ROOT, 'data', 'raw', 'V_Videos_Original', 'VIDEO2PREDICT_ORIGINAL')
VCAM_YOLO_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'vcam_yolo_classifier') # Nơi lưu model VCam gốc
VCAM_YOLO_MODEL_NAME = 'v_cam_yolo_model'
VCAM_YOLO_BEST_MODEL_SAVE_PATH = os.path.join(VCAM_YOLO_MODEL_DIR, f'{VCAM_YOLO_MODEL_NAME}_best.pt') # Model VCam GỐC
VCAM_YOLO_PRETRAINED_WEIGHTS_ORIGINAL = "yolo11n.pt"
VCAM_YOLO_IMG_SIZE = 640
VCAM_YOLO_EPOCHS_ORIGINAL = 100
VCAM_YOLO_BATCH_SIZE_ORIGINAL = 16
VCAM_YOLO_RUNS_DIR_ORIGINAL = os.path.join(PROJECT_ROOT, 'runs', 'train_vcam_original')
VCAM_YOLO_REPORTS_FIGURES_DIR_ORIGINAL = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'vcam_yolo_original')


# === Cấu hình cho Dữ liệu VCam từ YouTube (đã annotate thủ công, đã resplit) ===
VCAM_YOUTUBE_FINETUNE_DATA_SUBDIR_NAME = 'VCam_YouTube_FineTune_Data'
VCAM_YOUTUBE_FINETUNE_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed', VCAM_YOUTUBE_FINETUNE_DATA_SUBDIR_NAME)
VCAM_YOUTUBE_DATA_YAML_PATH = os.path.join(VCAM_YOUTUBE_FINETUNE_DATA_ROOT, 'data_vcam_youtube.yaml')

# === Cấu hình cho Mô hình VCam đã Fine-tune trên YouTube Data ===
VCAM_YOUTUBE_FINETUNED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'vcam_yolo_classifier_youtube_finetuned')
VCAM_YOUTUBE_FINETUNE_MODEL_NAME = 'v_cam_youtube_finetuned_model'
VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH = os.path.join(VCAM_YOUTUBE_FINETUNED_MODEL_DIR, f'{VCAM_YOUTUBE_FINETUNE_MODEL_NAME}_best.pt')
VCAM_YOUTUBE_FINETUNE_RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs', 'train_vcam_youtube')
VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'vcam_yolo_youtube_finetuned')
VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'vcam_yolo_youtube_finetuned') # <<< THÊM DÒNG NÀY
VCAM_YOUTUBE_FINETUNE_ANNOTATED_TEST_IMAGES_DIR = os.path.join(VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR, 'annotated_test_images')


# === Siêu tham số cho VCam YouTube Fine-tune ===
VCAM_YOUTUBE_FINETUNE_EPOCHS = 40
VCAM_YOUTUBE_FINETUNE_BATCH_SIZE = 8
VCAM_YOUTUBE_FINETUNE_IMG_SIZE = VCAM_YOLO_IMG_SIZE # Sử dụng lại imgsz từ config VCam gốc
VCAM_YOUTUBE_FINETUNE_LR0 = 1e-4
VCAM_YOUTUBE_FINETUNE_LRF = 1e-2
VCAM_YOUTUBE_FINETUNE_AUGMENT_PARAMS = {
    'mixup': 0.1, 'mosaic': 0.8, 'copy_paste': 0.05, 'hsv_h': 0.015, 'hsv_s': 0.7,
    'hsv_v': 0.4, 'degrees': 5.0, 'translate': 0.05, 'scale': 0.1, 'fliplr': 0.5
}


# === Cấu hình cho Dữ liệu Âm thanh từ YouTube (Dùng để Fine-tune Mô hình Audio) ===
# Thư mục chứa các video raw từ YouTube, được tổ chức theo các lớp trong MASTER_CLASS_LIST_FUSION
YOUTUBE_RAW_VIDEOS_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'YouTube_Videos_Raw')
# Thư mục chứa các segment audio .wav (1 giây) đã trích xuất từ video YouTube Audio_YouTube_FineTune_Segments
AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'Audio_YouTube_FineTune_Segments')
AUDIO_YOUTUBE_FINETUNE_METADATA_FILE = os.path.join(AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR, 'audio_youtube_finetune_metadata.csv')
# Các lớp sẽ được sử dụng để fine-tune mô hình audio trên dữ liệu YouTube
AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE = MASTER_CLASS_LIST_FUSION # Fine-tune cho tất cả các lớp của Fusion

# === Cấu hình cho Mô hình Âm thanh ĐÃ FINE-TUNE trên YouTube Data ===
AUDIO_YOUTUBE_FINETUNED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'audio_classifier_youtube_finetuned')
AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH = os.path.join(AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'best_audio_youtube_finetuned_model.keras')
AUDIO_YOUTUBE_LABEL_ENCODER_PATH = os.path.join(AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'label_encoder_audio_youtube.pkl')
AUDIO_YOUTUBE_SCALER_PATH = os.path.join(AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'scaler_audio_youtube.pkl')
AUDIO_YOUTUBE_MAX_LEN_PATH = os.path.join(AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'max_len_audio_youtube.pkl')
AUDIO_YOUTUBE_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'audio_youtube_finetuned')
AUDIO_YOUTUBE_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'audio_youtube_finetuned') # Thêm
AUDIO_YOUTUBE_TRAINING_HISTORY_PLOT_PATH = os.path.join(AUDIO_YOUTUBE_REPORTS_FIGURES_DIR, 'audio_youtube_ft_history.png')
AUDIO_YOUTUBE_CONFUSION_MATRIX_VAL_PLOT_PATH = os.path.join(AUDIO_YOUTUBE_REPORTS_FIGURES_DIR, 'audio_youtube_ft_cm_val.png')

# === Siêu tham số cho Fine-tuning Mô hình Âm thanh trên YouTube Data ===
AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE1 = 30 # 15
AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE2 = 40 # 25
AUDIO_YOUTUBE_FINETUNE_BATCH_SIZE = 16 # 32
AUDIO_YOUTUBE_FINETUNE_LR_STAGE1 = 1e-3
# AUDIO_YOUTUBE_FINETUNE_LR_STAGE2 = 5e-5
AUDIO_YOUTUBE_FINETUNE_LR_STAGE2 = 1e-6
AUDIO_YOUTUBE_VALIDATION_SPLIT = 0.20 #train 80% test 20


# === Cấu hình cho Dữ liệu Fine-tune Fusion (VCam + Audio) ===
FINETUNE_PAIRED_DATA_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'FineTune_Paired_Data')
FINETUNE_MULTICLASS_METADATA_FILE = os.path.join(FINETUNE_PAIRED_DATA_BASE_DIR, 'finetune_multiclass_metadata.csv')
INTERIM_YOUTUBE_AUDIO_DIR = os.path.join(PROJECT_ROOT, 'data', 'interim', 'YouTube_Audio_Extracted') # Dùng chung cho audio youtube fine-tune và fusion prep

FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'FineTune_Raw_Segments_For_Eval')
# QUAN TRỌNG: Chọn MÔ HÌNH AUDIO và VCAM nào để TRÍCH XUẤT FEATURE cho FUSION
# và để KHỞI TẠO NHÁNH AUDIO/VCAM trong FUSION MODEL
# Thay đổi các giá trị này sau khi đã fine-tune các model riêng lẻ trên YouTube data
AUDIO_MODEL_FOR_FUSION_BRANCH_INIT = AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH # SỬ DỤNG MODEL AUDIO ĐÃ FINE-TUNE TRÊN YOUTUBE
AUDIO_SCALER_FOR_FUSION_PREP = AUDIO_YOUTUBE_SCALER_PATH
AUDIO_MAX_LEN_FOR_FUSION_PREP = AUDIO_YOUTUBE_MAX_LEN_PATH

VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION = VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH # SỬ DỤNG MODEL VCAM ĐÃ FINE-TUNE TRÊN YOUTUBE

# === Cấu hình cho Mô hình Fusion ===
FUSION_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'fine_tuned_fusion_model')
BEST_FUSION_MODEL_SAVE_PATH = os.path.join(FUSION_MODEL_DIR, 'best_vcam_audio_fusion_model.keras')
FUSION_VCAM_FEATURE_DIM = 256
FUSION_NUM_CLASSES = len(MASTER_CLASS_LIST_FUSION)
FUSION_CLASS_NAMES = MASTER_CLASS_LIST_FUSION
FUSION_LABEL_ENCODER_PATH = os.path.join(FUSION_MODEL_DIR, 'label_encoder_fusion.pkl')
AUDIO_FEATURE_LAYER_NAME_FOR_FUSION = 'dropout_3' # HOẶC 'bidirectional_2' nếu bạn muốn output trực tiếp của BiLSTM
# AUDIO_FEATURE_LAYER_NAME_FOR_FUSION = 'bidirectional_2' # HOẶC 'bidirectional_2' nếu bạn muốn output trực tiếp của BiLSTM

# === Cấu hình Huấn luyện (Fine-tuning) cho Mô hình Fusion ===
FUSION_EPOCHS_STAGE1 = 40 # Dựa trên log trước của bạn
FUSION_EPOCHS_STAGE2 = 50 # Dựa trên log trước của bạn (hoặc số epoch thực tế EarlyStopping dừng)
FUSION_BATCH_SIZE = 16
FUSION_LEARNING_RATE_STAGE1 = 1e-4
# FUSION_LEARNING_RATE_STAGE2 = 5e-7 # Hoặc giá trị cuối cùng từ ReduceLROnPlateau
FUSION_LEARNING_RATE_STAGE2 = 1e-5 # thử kiểm nghiệm lại
FUSION_VALIDATION_SPLIT = 0.15

FUSION_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'fusion_vcam_audio')
FUSION_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'fusion_vcam_audio')
FUSION_TRAINING_HISTORY_STAGE1_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'fusion_training_history_stage1.png')
FUSION_TRAINING_HISTORY_STAGE2_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'fusion_training_history_stage2.png')
FUSION_CONFUSION_MATRIX_VAL_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'fusion_confusion_matrix_val.png')
FUSION_DETAILED_TEST_RESULTS_PATH = os.path.join(FUSION_REPORTS_METRICS_DIR, 'fusion_detailed_test_results.txt')
FUSION_VIDEOS_TO_PREDICT_DIR = os.path.join(YOUTUBE_RAW_VIDEOS_BASE_DIR, 'VIDEO2PREDICT_FUSION')

# === Cấu hình cho Mô hình Fusion (Tiếp theo) ===
# ... BEST_FUSION_MODEL_SAVE_PATH, FUSION_VCAM_FEATURE_DIM, etc. ...

# --- Đường dẫn lưu trữ DỮ LIỆU TEST IN-DOMAIN của FUSION ---
FUSION_TEST_IN_DOMAIN_DIR = os.path.join(FINETUNE_PAIRED_DATA_BASE_DIR, 'test_in_domain_for_fusion_eval')
# FUSION_TEST_IN_DOMAIN_DIR = os.path.join(FUSION_MODEL_DIR, 'test_in_domain_data') # Thư mục con
FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DIR, 'fusion_test_id_audio_features.pkl')
FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DIR, 'fusion_test_id_vcam_features.pkl')
FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DIR, 'fusion_test_id_true_labels.pkl')


# === Cấu hình Huấn luyện (Fine-tuning) cho Mô hình Fusion ===
FUSION_TEST_IN_DOMAIN_SPLIT_RATIO = 0.20 # 20% của dữ liệu fine-tune cho Test In-Domain
FUSION_VALIDATION_ON_REMAINING_RATIO = 0.20 # 20% của (100% - 20%) = 16% của tổng, cho Validation
# ... (FUSION_EPOCHS_STAGE1, etc.) ...

# Đường dẫn lưu BÁO CÁO cho Test In-Domain của Fusion
FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'fusion_vcam_audio_test_in_domain')
FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'fusion_vcam_audio_test_in_domain')
FUSION_CONFUSION_MATRIX_TEST_IN_DOMAIN_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, 'cm_fusion_test_in_domain.png')
FUSION_DETAILED_TEST_IN_DOMAIN_RESULTS_PATH = os.path.join(FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN, 'results_fusion_test_in_domain.txt')
# Thêm đường dẫn cho các mô hình đơn lẻ trên test in-domain
VCAM_YOLO_TEST_IN_DOMAIN_CM_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, 'cm_vcam_yolo_test_in_domain.png')
AUDIO_LSTM_TEST_IN_DOMAIN_CM_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, 'cm_audio_lstm_test_in_domain.png')

# Trong config/project_config.py
EVALUATION_TEST_RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'Final_Evaluation_Test_Set_Videos')
EVALUATION_TEST_SEGMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'Evaluation_Test_Segments')
EVALUATION_TEST_IMAGES_DIR = os.path.join(EVALUATION_TEST_SEGMENTS_BASE_DIR, 'images')
EVALUATION_TEST_AUDIO_SEGMENTS_DIR = os.path.join(EVALUATION_TEST_SEGMENTS_BASE_DIR, 'audio_segments')
EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE = os.path.join(EVALUATION_TEST_SEGMENTS_BASE_DIR, 'evaluation_test_ground_truth.csv')

EVALUATION_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'final_evaluation')
EVALUATION_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'final_evaluation')

# --- Đường dẫn lưu trữ DỮ LIỆU TEST IN-DOMAIN của FUSION (Features và Metadata) ---
FUSION_TEST_IN_DOMAIN_DATA_DIR = os.path.join(FINETUNE_PAIRED_DATA_BASE_DIR, 'test_in_domain_for_fusion_eval') # Thư mục con riêng
FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DATA_DIR, 'test_id_audio_features.pkl')
FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DATA_DIR, 'test_id_vcam_features.pkl')
FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DATA_DIR, 'test_id_true_labels.pkl')
FUSION_TEST_IN_DOMAIN_METADATA_PATH = os.path.join(FUSION_TEST_IN_DOMAIN_DATA_DIR, 'test_id_metadata.csv') # Metadata cho Test In-Domain

# --- Đường dẫn báo cáo cho Fusion Model ---
FUSION_REPORTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'fusion_vcam_audio')
FUSION_REPORTS_METRICS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'metrics', 'fusion_vcam_audio')
FUSION_TRAINING_HISTORY_STAGE1_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'fusion_training_history_stage1.png')
FUSION_TRAINING_HISTORY_STAGE2_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'fusion_training_history_stage2.png')
FUSION_CONFUSION_MATRIX_VAL_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'fusion_confusion_matrix_val.png')
# Báo cáo cho Test In-Domain của Fusion
FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN = os.path.join(FUSION_REPORTS_FIGURES_DIR, 'test_in_domain') # Thư mục con
FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN = os.path.join(FUSION_REPORTS_METRICS_DIR, 'test_in_domain') # Thư mục con
FUSION_CONFUSION_MATRIX_TEST_IN_DOMAIN_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, 'cm_fusion_test_in_domain.png')
# ... (Các đường dẫn báo cáo khác cho VCam YOLO và Audio LSTM trên Test In-Domain sẽ được tạo động trong script eval)

# === Cấu hình cho Đánh giá Cuối cùng trên Tập Test OOD ===


EVALUATION_OOD_ANNOTATED_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'ood_annotated_images_vcam') # <<< THÊM DÒNG NÀY
EVALUATION_OOD_FRAMES_FOR_ROBOFLOW_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'OOD_Frames_For_Roboflow') # <<< THÊM DÒNG NÀY

# Đường dẫn cụ thể cho CM và report của VCam trên Test In-Domain
VCAM_YOLO_TEST_IN_DOMAIN_REPORT_PATH = os.path.join(FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN, 'report_vcam_yolo_test_in_domain.txt') # <<< THÊM DÒNG NÀY

# Đường dẫn cụ thể cho CM và report của Audio LSTM trên Test In-Domain
AUDIO_LSTM_TEST_IN_DOMAIN_REPORT_PATH = os.path.join(FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN, 'report_audio_lstm_test_in_domain.txt') # <<< THÊM DÒNG NÀY

# Đường dẫn cụ thể cho CM và report của Fusion Model trên Test In-Domain
FUSION_MODEL_TEST_IN_DOMAIN_CM_PLOT_PATH = os.path.join(FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, 'cm_fusion_model_test_in_domain.png')
FUSION_MODEL_TEST_IN_DOMAIN_REPORT_PATH = os.path.join(FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN, 'report_fusion_model_test_in_domain.txt') # (Đã có từ trước)


# --- Đảm bảo các thư mục lưu trữ tồn tại ---
def ensure_output_directories():
    dirs_to_create = [
        # Audio Gốc
        AUDIO_MODEL_DIR, AUDIO_PROCESSED_DATA_DIR,
        AUDIO_REPORTS_FIGURES_DIR_ORIGINAL, AUDIO_REPORTS_METRICS_DIR_ORIGINAL,

        # Audio YouTube Fine-tune
        AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR,
        AUDIO_YOUTUBE_FINETUNED_MODEL_DIR,
        AUDIO_YOUTUBE_REPORTS_FIGURES_DIR, AUDIO_YOUTUBE_REPORTS_METRICS_DIR,


        # IRCam YOLO
        IRCAM_YOLO_MODEL_DIR, IRCAM_PROCESSED_DATA_FULL_PATH,
        IRCAM_VIDEOS_TO_PREDICT_DIR, IRCAM_YOLO_REPORTS_FIGURES_DIR,
        IRCAM_YOLO_RUNS_DIR,

        # VCam YOLO (Gốc)
        VCAM_YOLO_MODEL_DIR, VCAM_PROCESSED_DATA_FULL_PATH_ORIGINAL,
        VCAM_VIDEOS_TO_PREDICT_DIR_ORIGINAL, VCAM_YOLO_REPORTS_FIGURES_DIR_ORIGINAL,
        VCAM_YOLO_RUNS_DIR_ORIGINAL,

        # VCam YouTube Fine-tune (YOLO)
        VCAM_YOUTUBE_FINETUNE_DATA_ROOT,
        VCAM_YOUTUBE_FINETUNED_MODEL_DIR,
        VCAM_YOUTUBE_FINETUNE_RUNS_DIR,
        VCAM_YOUTUBE_FINETUNE_REPORTS_FIGURES_DIR,
        VCAM_YOUTUBE_FINETUNE_REPORTS_METRICS_DIR,
        VCAM_YOUTUBE_FINETUNE_ANNOTATED_TEST_IMAGES_DIR,
        # Fusion Fine-tune
        YOUTUBE_RAW_VIDEOS_BASE_DIR,
        FINETUNE_PAIRED_DATA_BASE_DIR,
        INTERIM_YOUTUBE_AUDIO_DIR,
        FUSION_MODEL_DIR,
        FUSION_REPORTS_FIGURES_DIR,
        FUSION_REPORTS_METRICS_DIR,
        FUSION_VIDEOS_TO_PREDICT_DIR,
        
        # === THÊM CÁC THƯ MỤC CHO FINAL EVALUATION TEST SET ===
        EVALUATION_TEST_RAW_VIDEO_DIR,
        EVALUATION_TEST_SEGMENTS_BASE_DIR, # Thư mục cha cho segments
        EVALUATION_TEST_IMAGES_DIR,         # Thư mục con images
        EVALUATION_TEST_AUDIO_SEGMENTS_DIR, # Thư mục con audio_segments
        EVALUATION_REPORTS_FIGURES_DIR,     # Thư mục figures cho final evaluation
        EVALUATION_REPORTS_METRICS_DIR,      # Thư mục metrics cho final evaluation
        
        FUSION_TEST_IN_DOMAIN_DIR, # Thư mục lưu features test in-domain
        FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN,
        FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN,
        
        # Fusion Fine-tune
        YOUTUBE_RAW_VIDEOS_BASE_DIR, FINETUNE_PAIRED_DATA_BASE_DIR, INTERIM_YOUTUBE_AUDIO_DIR,
        FUSION_MODEL_DIR, FUSION_REPORTS_FIGURES_DIR, FUSION_REPORTS_METRICS_DIR,
        FUSION_VIDEOS_TO_PREDICT_DIR, FUSION_TEST_IN_DOMAIN_DATA_DIR, # Sửa tên biến
        FUSION_REPORTS_FIGURES_DIR_TEST_IN_DOMAIN, FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN,
        # Final OOD Evaluation
        
        
        FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR,
        EVALUATION_OOD_ANNOTATED_IMAGES_DIR,
        EVALUATION_OOD_FRAMES_FOR_ROBOFLOW_DIR,
        
    ]
    # Tạo thư mục con cho từng lớp trong YOUTUBE_RAW_VIDEOS_BASE_DIR,
    # FINETUNE_PAIRED_DATA_BASE_DIR, và AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR
    for class_name in MASTER_CLASS_LIST_FUSION: # Dùng danh sách lớp chung cho fusion
        dirs_to_create.append(os.path.join(YOUTUBE_RAW_VIDEOS_BASE_DIR, class_name))
        dirs_to_create.append(os.path.join(FINETUNE_PAIRED_DATA_BASE_DIR, class_name))
        dirs_to_create.append(os.path.join(AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR, class_name))
        dirs_to_create.append(os.path.join(EVALUATION_TEST_RAW_VIDEO_DIR, class_name))
        
        dirs_to_create.append(os.path.join(FINETUNE_RAW_SEGMENTS_FOR_EVAL_DIR, class_name)) # <<< THÊM
        
    for d_path in dirs_to_create:
        os.makedirs(d_path, exist_ok=True)
    # print("Output directories checked/created from project_config.py (during import).")

# Tự động chạy khi module này được import lần đầu tiên (nếu không phải là script chính đang chạy)
if __name__ != "__main__":
    # print(f"project_config.py loaded. Ensuring output directories...")
    ensure_output_directories()