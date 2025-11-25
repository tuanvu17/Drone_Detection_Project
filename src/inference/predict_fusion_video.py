# Drone_Detection_Project/src/inference/predict_fusion_video.py
import os
# BUỘC SỬ DỤNG CPU CHO TENSORFLOW/KERAS NGAY TỪ ĐẦU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import glob
import cv2
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from ultralytics import YOLO
from moviepy.video.io.VideoFileClip import VideoFileClip as MoviePyVideoFileClip
from tqdm import tqdm
import time
import tensorflow as tf

try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input
except ImportError:
    import sys
    current_dir_inf = os.path.dirname(os.path.abspath(__file__))
    src_dir_inf = os.path.dirname(current_dir_inf)
    project_root_inf = os.path.dirname(src_dir_inf)
    if project_root_inf not in sys.path: sys.path.insert(0, project_root_inf)
    if src_dir_inf not in sys.path: sys.path.insert(0, src_dir_inf)
    from config_loader.loader import get_config
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from data_processing.prepare_fusion_finetune_data import FeatureExtractorVCam, preprocess_frame_for_yolo_input

cfg = get_config()

def predict_on_single_fusion_video_with_boxes( # Đổi tên hàm để rõ ràng hơn
                                   video_path,
                                   fusion_model_keras,
                                   vcam_yolo_detector_torch, # << THÊM: Model YOLO để detect box
                                   vcam_feature_extractor_torch,
                                   audio_scaler,
                                   audio_max_len,
                                   fusion_label_encoder,
                                   device_torch, # Sẽ là 'cpu'
                                   output_video_path=None,
                                   yolo_conf_thresh=0.3): # Ngưỡng conf cho YOLO detector
    print(f"\n--- Đang xử lý video cho Fusion Prediction (with BBoxes): {os.path.basename(video_path)} ---")
    print(f"    Sử dụng thiết bị PyTorch: {device_torch}")

    # --- Trích xuất Audio ---
    audio_data = None
    temp_audio_path = os.path.join(cfg.INTERIM_YOUTUBE_AUDIO_DIR, f"{os.path.splitext(os.path.basename(video_path))[0]}_pred_audio.wav")
    os.makedirs(cfg.INTERIM_YOUTUBE_AUDIO_DIR, exist_ok=True)
    try:
        with MoviePyVideoFileClip(video_path) as video_clip:
            if video_clip.audio is None:
                print(f"  CẢNH BÁO: Video {os.path.basename(video_path)} không có track audio. Sử dụng tín hiệu im lặng.")
                cap_temp_duration = cv2.VideoCapture(video_path)
                if cap_temp_duration.isOpened():
                    fps_temp = cap_temp_duration.get(cv2.CAP_PROP_FPS)
                    frame_count_temp = cap_temp_duration.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps_temp > 0 and frame_count_temp > 0:
                        video_actual_duration = frame_count_temp / fps_temp
                        num_audio_samples_for_duration = int(video_actual_duration * cfg.AUDIO_SAMPLE_RATE)
                        audio_data = np.zeros(num_audio_samples_for_duration, dtype=np.float32)
                    cap_temp_duration.release()
                if audio_data is None: audio_data = np.zeros(int(cfg.AUDIO_SEGMENT_DURATION * 5 * cfg.AUDIO_SAMPLE_RATE), dtype=np.float32)
            else:
                video_clip.audio.write_audiofile(temp_audio_path, fps=cfg.AUDIO_SAMPLE_RATE, logger=None)
                audio_data, _ = librosa.load(temp_audio_path, sr=cfg.AUDIO_SAMPLE_RATE)
    except Exception as e:
        print(f"  Lỗi xử lý audio từ video: {e}. Sử dụng tín hiệu im lặng.")
        if audio_data is None: audio_data = np.zeros(int(cfg.AUDIO_SEGMENT_DURATION * 5 * cfg.AUDIO_SAMPLE_RATE), dtype=np.float32)
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
    if audio_data is None or len(audio_data) == 0: return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"  Lỗi mở video {os.path.basename(video_path)}."); return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 25
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_cv = total_frames_video / fps if fps > 0 else 0
    max_segments_audio = len(audio_data) // int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
    max_segments_video = int(video_duration_cv // cfg.AUDIO_SEGMENT_DURATION) if cfg.AUDIO_SEGMENT_DURATION > 0 else 0
    num_segments = min(max_segments_audio, max_segments_video)
    if num_segments == 0: cap.release(); print(f"Video quá ngắn: {os.path.basename(video_path)}"); return None

    all_segment_fusion_probs = []
    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (vid_width, vid_height))
        print(f"  Video kết quả sẽ được lưu tại: {output_video_path}")

    print(f"  Xử lý {num_segments} segment(s)...")
    current_frame_read_index = 0

    for i in tqdm(range(num_segments), desc=f"Segments in {os.path.basename(video_path)}"):
        # --- Xử lý Audio cho segment ---
        audio_start_sample = int(i * cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
        audio_end_sample = int((i + 1) * cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
        audio_segment = audio_data[audio_start_sample:min(audio_end_sample, len(audio_data))]
        if len(audio_segment) < int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.5):
            # ... (Logic ghi frame gốc vào writer nếu segment audio lỗi, cập nhật current_frame_read_index) ...
            if writer:
                for _ in range(int(round(fps * cfg.AUDIO_SEGMENT_DURATION))):
                    if current_frame_read_index < total_frames_video:
                        # Không cần cap.set() ở đây nếu đọc tuần tự
                        ret_f, frame_f = cap.read() # Đọc frame tiếp theo
                        if ret_f: writer.write(frame_f)
                        current_frame_read_index +=1
                    else: break
            continue
        try:
            mfcc_features = extract_mfcc(audio_segment, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
            mfcc_padded_single, _ = pad_features([mfcc_features], max_len=audio_max_len)
            if mfcc_padded_single.size == 0: # ... (xử lý lỗi và ghi frame gốc) ...
                 if writer:
                    for _ in range(int(round(fps * cfg.AUDIO_SEGMENT_DURATION))):
                        if current_frame_read_index < total_frames_video:
                            ret_f, frame_f = cap.read()
                            if ret_f: writer.write(frame_f)
                            current_frame_read_index +=1
                        else: break
                 continue
            mfcc_scaled_single, _ = scale_features(mfcc_padded_single, scaler=audio_scaler)
            processed_audio_feature = mfcc_scaled_single # Shape (1, max_len, n_mfcc)
        except Exception as e_audio:
            print(f"    Lỗi xử lý audio cho segment {i}: {e_audio}")
            # ... (Logic ghi frame gốc vào writer nếu segment audio lỗi, cập nhật current_frame_read_index) ...
            if writer:
                for _ in range(int(round(fps * cfg.AUDIO_SEGMENT_DURATION))):
                    if current_frame_read_index < total_frames_video:
                        ret_f, frame_f = cap.read()
                        if ret_f: writer.write(frame_f)
                        current_frame_read_index +=1
                    else: break
            continue

        # --- Xử lý VCam Frame cho segment ---
        # Lấy frame đại diện cho việc trích xuất VCam feature (cho fusion)
        frame_num_target_for_vcam_features = int(((i + 0.5) * cfg.AUDIO_SEGMENT_DURATION) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_target_for_vcam_features)
        ret_vcam_feat, frame_vcam_for_features = cap.read()

        processed_vcam_feature_for_fusion = None # Khởi tạo
        if not ret_vcam_feat or frame_vcam_for_features is None:
            print(f"    Cảnh báo: Không đọc được VCam frame cho feature extraction segment {i}. Sử dụng VCam feature giả.")
            dummy_vcam_feature = np.zeros((1, cfg.FUSION_VCAM_FEATURE_DIM), dtype=np.float32) # Thêm batch dim
            processed_vcam_feature_for_fusion = dummy_vcam_feature
        else:
            try:
                vcam_input_tensor = preprocess_frame_for_yolo_input(frame_vcam_for_features, cfg.VCAM_YOLO_IMG_SIZE) # Dùng VCAM_YOLO_IMG_SIZE cho feature extractor
                with torch.no_grad():
                    vcam_feature_vector = vcam_feature_extractor_torch(vcam_input_tensor.to(device_torch)).cpu().numpy()
                processed_vcam_feature_for_fusion = vcam_feature_vector # Đã có batch_dim (1, feature_dim)
            except Exception as e_vcam_feat:
                print(f"    Lỗi trích xuất VCam feature cho segment {i}: {e_vcam_feat}. Sử dụng VCam feature giả.")
                dummy_vcam_feature = np.zeros((1, cfg.FUSION_VCAM_FEATURE_DIM), dtype=np.float32)
                processed_vcam_feature_for_fusion = dummy_vcam_feature

        # --- Dự đoán bằng Mô hình Fusion Keras ---
        segment_fusion_probs = None
        predicted_class_name_fusion = "N/A"
        confidence_fusion = 0.0
        try:
            # Cả hai input đều phải có batch dimension
            segment_fusion_probs = fusion_model_keras.predict([processed_audio_feature, processed_vcam_feature_for_fusion], verbose=0)[0]
            all_segment_fusion_probs.append(segment_fusion_probs)
            predicted_class_idx_fusion = np.argmax(segment_fusion_probs)
            predicted_class_name_fusion = fusion_label_encoder.classes_[predicted_class_idx_fusion]
            confidence_fusion = segment_fusion_probs[predicted_class_idx_fusion]
        except Exception as e_fusion_pred:
            print(f"    Lỗi khi dự đoán bằng fusion model cho segment {i}: {e_fusion_pred}")
            # Không thêm vào all_segment_fusion_probs nếu lỗi

        # --- Ghi Video Output với Bounding Box từ VCam YOLO Detector và Text từ Fusion ---
        if writer:
            text_fusion_to_draw = f"Fusion: {predicted_class_name_fusion} ({confidence_fusion:.2f})"
            num_frames_in_segment_actual = int(round(fps * cfg.AUDIO_SEGMENT_DURATION))

            for k_frame in range(num_frames_in_segment_actual):
                if current_frame_read_index < total_frames_video:
                    # Đọc frame gốc tuần tự từ video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_read_index) # Đảm bảo đọc đúng frame
                    ret_write, frame_to_annotate = cap.read()
                    if ret_write:
                        # 1. Chạy VCam YOLO Detector trên frame hiện tại để lấy boxes
                        yolo_detection_results = vcam_yolo_detector_torch.predict(
                            source=frame_to_annotate.copy(),
                            imgsz=cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE, # Dùng imgsz của detector
                            conf=yolo_conf_thresh,
                            verbose=False,
                            device=device_torch
                        )
                        # Vẽ box từ YOLO
                        if yolo_detection_results and yolo_detection_results[0].boxes:
                            # frame_to_annotate = yolo_detection_results[0].plot() # Cách này sẽ ghi đè
                            # Vẽ thủ công để giữ frame gốc và thêm text fusion
                            for box_data in yolo_detection_results[0].boxes:
                                x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
                                conf_y = float(box_data.conf[0])
                                cls_id_y = int(box_data.cls[0])
                                cls_name_y = vcam_yolo_detector_torch.names[cls_id_y] # Lấy tên từ model detector
                                label_yolo = f"{cls_name_y}: {conf_y:.2f}"
                                cv2.rectangle(frame_to_annotate, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame_to_annotate, label_yolo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # 2. Vẽ text dự đoán của Fusion Model
                        cv2.putText(frame_to_annotate, text_fusion_to_draw, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                        writer.write(frame_to_annotate)
                    current_frame_read_index += 1
                else: break # Hết video gốc
        else: # Nếu không ghi video, vẫn phải tăng current_frame_read_index
            current_frame_read_index += int(round(fps * cfg.AUDIO_SEGMENT_DURATION))
            if current_frame_read_index > total_frames_video : current_frame_read_index = total_frames_video


    cap.release()
    if writer:
        writer.release()
        print(f"  Đã lưu video kết quả vào: {output_video_path}")

    if not all_segment_fusion_probs:
        print(f"  Không có dự đoán segment nào được thực hiện cho video {os.path.basename(video_path)}.")
        return None

    final_video_avg_probs = np.mean(np.array(all_segment_fusion_probs), axis=0)
    final_pred_idx = np.argmax(final_video_avg_probs)
    final_pred_class = fusion_label_encoder.classes_[final_pred_idx]
    final_confidence = float(final_video_avg_probs[final_pred_idx])

    print(f"  Dự đoán cuối cùng cho video '{os.path.basename(video_path)}': {final_pred_class} (Confidence: {final_confidence:.4f})")
    return {
        'video_path': video_path,
        'predicted_class': final_pred_class,
        'confidence': final_confidence,
        'all_class_probabilities': final_video_avg_probs.tolist(),
        'all_class_names': list(fusion_label_encoder.classes_)
    }


if __name__ == "__main__":
    print("Thiết lập chạy trên CPU cho TensorFlow...")
    if tf.config.list_physical_devices('GPU'):
        print("CẢNH BÁO: GPU vẫn được TensorFlow phát hiện.")
    else:
        print("TensorFlow sẽ chạy trên CPU.")
    device_torch_main = torch.device("cpu")
    print(f"Sử dụng thiết bị PyTorch: {device_torch_main}")

    if hasattr(cfg, 'ensure_output_directories'): cfg.ensure_output_directories()
    print("Đang tải các mô hình và thành phần tiền xử lý...")
    try:
        fusion_model_keras_main = tf.keras.models.load_model(cfg.BEST_FUSION_MODEL_SAVE_PATH)
        print(f"  Đã tải Fusion Model Keras từ: {cfg.BEST_FUSION_MODEL_SAVE_PATH}")

        # Tải VCam YOLO model đã fine-tune trên YouTube để PHÁT HIỆN BOX
        vcam_yolo_detector_main = YOLO(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        print(f"  Đã tải VCam YOLO Detector (fine-tuned on YouTube) từ: {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH}")

        # Tải VCam model được dùng để TRÍCH XUẤT FEATURE khi huấn luyện fusion model
        vcam_yolo_for_feat_ext_path = cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION
        print(f"  Đang tải VCam model for feature extraction từ: {vcam_yolo_for_feat_ext_path}")
        vcam_yolo_for_feature_extraction = YOLO(vcam_yolo_for_feat_ext_path)
        vcam_feat_extractor_main = FeatureExtractorVCam(
            vcam_yolo_for_feature_extraction.model.model,
            extraction_layer_index=8 # Index 8 cho tầng C3k2 cuối backbone
        )
        vcam_feat_extractor_main.to(device_torch_main)
        vcam_feat_extractor_main.eval()
        print(f"  Đã tải và thiết lập VCam Feature Extractor trên CPU.")

        with open(cfg.AUDIO_SCALER_PATH, 'rb') as f: audio_scaler_main = pickle.load(f)
        with open(cfg.AUDIO_MAX_LEN_PATH, 'rb') as f: audio_max_len_main = pickle.load(f)
        with open(cfg.FUSION_LABEL_ENCODER_PATH, 'rb') as f: fusion_le_main = pickle.load(f)
        print("  Đã tải Audio Scaler, MaxLen, và Fusion LabelEncoder.")
    except Exception as e:
        print(f"LỖI: Không thể tải các thành phần cần thiết cho dự đoán: {e}")
        import traceback
        traceback.print_exc()
        exit()

    videos_to_predict_input_dir = cfg.FUSION_VIDEOS_TO_PREDICT_DIR
    os.makedirs(videos_to_predict_input_dir, exist_ok=True)
    print(f"\nSẽ tìm video để dự đoán trong thư mục: {videos_to_predict_input_dir}")

    video_files_to_predict_main = glob.glob(os.path.join(videos_to_predict_input_dir, "*.*"))
    video_files_to_predict_main = [f for f in video_files_to_predict_main if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files_to_predict_main:
        print(f"Không tìm thấy video nào trong thư mục: {videos_to_predict_input_dir}")
    else:
        print(f"\n--- Bắt đầu dự đoán Fusion trên {len(video_files_to_predict_main)} video ---")
        all_results_main = []
        for video_fpath_main in video_files_to_predict_main:
            base_name, ext = os.path.splitext(os.path.basename(video_fpath_main))
            # Đảm bảo thư mục output cho video này tồn tại
            os.makedirs(cfg.FUSION_REPORTS_FIGURES_DIR, exist_ok=True)
            output_vid_path_main = os.path.join(cfg.FUSION_REPORTS_FIGURES_DIR, f"{base_name}_fusion_predicted_with_boxes{ext}")

            result = predict_on_single_fusion_video_with_boxes(
                video_path=video_fpath_main,
                fusion_model_keras=fusion_model_keras_main,
                vcam_yolo_detector_torch=vcam_yolo_detector_main, # Truyền model YOLO để detect box
                vcam_feature_extractor_torch=vcam_feat_extractor_main,
                audio_scaler=audio_scaler_main,
                audio_max_len=audio_max_len_main,
                fusion_label_encoder=fusion_le_main,
                device_torch=device_torch_main,
                output_video_path=output_vid_path_main,
                yolo_conf_thresh=0.3
            )
            if result:
                all_results_main.append(result)

        print("\n\n--- TÓM TẮT KẾT QUẢ DỰ ĐOÁN FUSION ---")
        for res_main in all_results_main:
            print(f"File: {os.path.basename(res_main['video_path'])}")
            print(f"  Lớp dự đoán: {res_main['predicted_class']}")
            print(f"  Độ tự tin: {res_main['confidence']:.4f}")
            print(f"  Xác suất các lớp ({res_main['all_class_names']}):")
            for class_n, prob_n in zip(res_main['all_class_names'], res_main['all_class_probabilities']):
                print(f"    - {class_n}: {prob_n:.4f}")
            print("-" * 30)
    print("--- Dự đoán video bằng mô hình Fusion (with BBoxes) hoàn tất ---")