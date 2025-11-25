# Drone_Detection_Project/src/main_evaluate_comparison_in_domain.py
import os
import sys
import pickle # Để nối các report text

# Đặt os.environ TRƯỚC khi import TensorFlow (nếu có)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Buộc TensorFlow chạy trên CPU

PROJECT_ROOT_DIR_MAIN_EVAL_ID = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR_MAIN_EVAL_ID not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR_MAIN_EVAL_ID)

try:
    from src.evaluation.evaluate_vcam_yolo_on_test_in_domain import run_vcam_yolo_evaluation_on_test_in_domain
    from src.evaluation.evaluate_audio_lstm_on_test_in_domain import run_audio_lstm_evaluation_on_test_in_domain
    from src.evaluation.evaluate_fusion_on_test_in_domain import run_fusion_evaluation_on_test_in_domain # MỚI
    from src.config_loader.loader import get_config
except ImportError as e:
    print(f"Lỗi khi import các module cần thiết: {e}"); exit()

def main():
    print("===== BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH TRÊN TEST IN-DOMAIN =====")
    cfg = get_config()
    if hasattr(cfg, 'ensure_output_directories'): cfg.ensure_output_directories()

    # Kiểm tra các file đầu vào chính cần thiết
    required_files = [
        cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH, # Metadata của Test In-Domain (chứa đường dẫn file gốc)
        cfg.FUSION_LABEL_ENCODER_PATH,           # Encoder chung (fit trên FUSION_CLASS_NAMES)
        
        cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH, # Model VCam
        
        cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH,  # Model Audio fine-tuned
        cfg.AUDIO_YOUTUBE_SCALER_PATH,
        cfg.AUDIO_YOUTUBE_MAX_LEN_PATH,
        
        cfg.BEST_FUSION_MODEL_SAVE_PATH,              # Model Fusion
        # Các file features cho fusion test in-domain (đã được lưu bởi fine_tune_fusion_pipeline)
        cfg.FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH,
        cfg.FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH,
        cfg.FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH,
    ]
    print("\nKiểm tra các file đầu vào cần thiết cho pipeline so sánh model trên Test In-Domain:")
    all_exist = True
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"  LỖI: Thiếu file đầu vào: {f_path}"); all_exist = False
        else: print(f"  OK: {f_path}")
    if not all_exist: print("\nThiếu file. Dừng."); return

    # Chạy trên CPU cho các model PyTorch
    pytorch_device = 'cpu'

    # 1. Đánh giá VCam YOLO (YouTube Fine-tuned) trên Test In-Domain
    try:
        print("\n\n" + "="*20 + " EVALUATING VCAM YOLO ON TEST IN-DOMAIN " + "="*20)
        run_vcam_yolo_evaluation_on_test_in_domain()
    except Exception as e_vcam_eval: print(f"LỖI VCam YOLO Eval: {e_vcam_eval}")

    # 2. Đánh giá Audio LSTM (YouTube Fine-tuned) trên Test In-Domain
    try:
        print("\n\n" + "="*20 + " EVALUATING AUDIO LSTM ON TEST IN-DOMAIN " + "="*20)
        run_audio_lstm_evaluation_on_test_in_domain() # TF đã được set CPU
    except Exception as e_audio_eval: print(f"LỖI Audio LSTM Eval: {e_audio_eval}")

    # 3. Đánh giá Fusion Model trên Test In-Domain (sử dụng features đã lưu)
    try:
        print("\n\n" + "="*20 + " EVALUATING FUSION MODEL ON TEST IN-DOMAIN " + "="*20)
        run_fusion_evaluation_on_test_in_domain() # TF đã được set CPU
    except Exception as e_fusion_eval: print(f"LỖI Fusion Model Eval: {e_fusion_eval}")

    # (Tùy chọn) Gộp các file report text lại
    print("\n--- Gộp các báo cáo ---")
    combined_report_path = os.path.join(cfg.FUSION_REPORTS_METRICS_DIR_TEST_IN_DOMAIN, 'all_models_comparison_report_test_in_domain.txt')
    report_files_to_combine = [
        cfg.VCAM_YOLO_TEST_IN_DOMAIN_REPORT_PATH,
        cfg.AUDIO_LSTM_TEST_IN_DOMAIN_REPORT_PATH,
        cfg.FUSION_MODEL_TEST_IN_DOMAIN_REPORT_PATH
    ]
    with open(combined_report_path, 'w') as outfile:
        for fname in report_files_to_combine:
            if os.path.exists(fname):
                with open(fname, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n" + "="*70 + "\n\n")
            else:
                outfile.write(f"Báo cáo cho {fname} không tìm thấy.\n\n" + "="*70 + "\n\n")
    print(f"Báo cáo so sánh tổng hợp đã lưu vào: {combined_report_path}")


    print("\n===== QUY TRÌNH ĐÁNH GIÁ SO SÁNH CÁC MÔ HÌNH TRÊN TEST IN-DOMAIN HOÀN TẤT =====")

if __name__ == "__main__":
    main()