# Drone_Detection_Project/src/data_processing/prepare_fusion_finetune_data.py
import os
import glob
import cv2
import librosa
import numpy as np
import pickle
import torch
import torch.nn as nn
from ultralytics import YOLO
from moviepy.video.io.VideoFileClip import VideoFileClip # Sử dụng import đã sửa
import pandas as pd
from tqdm import tqdm
import time
import soundfile as sf # Để lưu audio segment gốc

# Import từ các module trong src
try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
except ImportError:
    import sys
    current_dir_dp = os.path.dirname(os.path.abspath(__file__))
    src_dir_dp = os.path.dirname(current_dir_dp)
    project_root_dp = os.path.dirname(src_dir_dp)
    if project_root_dp not in sys.path: sys.path.insert(0, project_root_dp)
    if src_dir_dp not in sys.path: sys.path.insert(0, src_dir_dp)
    from config_loader.loader import get_config
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features

cfg = get_config()
device_torch_prep = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Định nghĩa VCam Feature Extractor (PyTorch) ---
class FeatureExtractorVCam(nn.Module):
    def __init__(self, original_yolo_model_sequential, extraction_layer_index):
        super().__init__()
        self.features_extractor_sequential = nn.Sequential(*list(original_yolo_model_sequential.children())[:extraction_layer_index + 1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_features = self.features_extractor_sequential(x)
        x_pooled = self.pool(x_features)
        x_flattened = self.flatten(x_pooled)
        return x_flattened

def preprocess_frame_for_yolo_input(frame_bgr, target_img_size_yolo):
    if frame_bgr is None:
        print("Lỗi (preprocess_frame_for_yolo_input): frame_bgr đầu vào là None.")
        return None
    img_h, img_w = frame_bgr.shape[:2]
    if img_h == 0 or img_w == 0:
        print(f"Lỗi (preprocess_frame_for_yolo_input): Kích thước frame không hợp lệ: H={img_h}, W={img_w}")
        return None

    new_w, new_h = target_img_size_yolo, target_img_size_yolo
    scale = min(new_w / img_w, new_h / img_h)
    resized_w, resized_h = int(round(img_w * scale)), int(round(img_h * scale))

    if resized_w == 0 or resized_h == 0: # Kiểm tra sau khi tính scale
        print(f"Lỗi (preprocess_frame_for_yolo_input): Kích thước sau resize không hợp lệ: Resized_W={resized_w}, Resized_H={resized_h} từ scale={scale}")
        return None

    if img_w != resized_w or img_h != resized_h:
        frame_resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame_bgr.copy()

    dw, dh = (new_w - resized_w), (new_h - resized_h)
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if (top + bottom) < (new_h - resized_h): bottom +=1
    if (left + right) < (new_w - resized_w): right +=1

    img_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
    img_tensor /= 255.0
    return img_tensor.unsqueeze(0)


def prepare_fine_tune_data_multiclass():
    print("===== STARTING PREPARATION OF MULTI-CLASS FINE-TUNE DATA (VCAM + AUDIO) =====")
    if hasattr(cfg, 'ensure_output_directories'):
        cfg.ensure_output_directories()

    print("Đang tải Scaler và MaxLen cho Audio (từ Audio YouTube Fine-tune)...")
    try:
        with open(cfg.AUDIO_SCALER_FOR_FUSION_PREP, 'rb') as f:
            audio_scaler_for_fusion = pickle.load(f)
        with open(cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP, 'rb') as f:
            audio_max_len_for_fusion = pickle.load(f)
        print(f"  Scaler và MaxLen Audio cho Fusion đã tải thành công. MaxLen: {audio_max_len_for_fusion}")
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file Scaler ({cfg.AUDIO_SCALER_FOR_FUSION_PREP}) hoặc MaxLen ({cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP}). Lỗi: {e}")
        print("Vui lòng chạy pipeline fine-tune audio trên YouTube trước để tạo các file này.")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi tải audio scaler/maxlen cho fusion: {e}"); return

    print("Đang tải mô hình VCam YOLO để trích xuất đặc trưng (từ fine-tune trên YouTube)...")
    try:
        vcam_yolo_full_model_for_extraction = YOLO(cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION)
        if hasattr(vcam_yolo_full_model_for_extraction, 'model') and hasattr(vcam_yolo_full_model_for_extraction.model, 'model'):
            yolo_sequential_part = vcam_yolo_full_model_for_extraction.model.model
        else:
            raise ValueError("Không thể truy cập phần sequential của mô hình YOLO. Kiểm tra cấu trúc model đã tải.")
        EXTRACTION_LAYER_IDX = 8
        vcam_feature_extractor = FeatureExtractorVCam(yolo_sequential_part, EXTRACTION_LAYER_IDX)
        vcam_feature_extractor.to(device_torch_prep).eval()
        print(f"  Sử dụng thiết bị: {device_torch_prep} cho VCam feature extraction.")
        print(f"  Trích xuất feature VCam từ lớp {EXTRACTION_LAYER_IDX} của backbone model: {cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION}")
    except Exception as e:
        print(f"LỖI: Không thể tải hoặc thiết lập VCam feature extractor: {e}"); return

    metadata_list = []
    global_segment_counter = 0
    raw_segments_output_base_dir = os.path.join(cfg.PROJECT_ROOT, 'data', 'processed', 'FineTune_Raw_Segments_For_Eval')
    os.makedirs(raw_segments_output_base_dir, exist_ok=True)

    for class_name in cfg.MASTER_CLASS_LIST_FUSION:
        print(f"\n--- Processing class: {class_name} ---")
        raw_video_dir_for_class = os.path.join(cfg.YOUTUBE_RAW_VIDEOS_BASE_DIR, class_name)
        if not os.path.isdir(raw_video_dir_for_class):
            print(f"  Cảnh báo: Không tìm thấy thư mục video raw cho lớp '{class_name}'. Bỏ qua."); continue

        output_feature_class_dir = os.path.join(cfg.FINETUNE_PAIRED_DATA_BASE_DIR, class_name)
        output_raw_segment_class_dir = os.path.join(raw_segments_output_base_dir, class_name)
        os.makedirs(output_feature_class_dir, exist_ok=True)
        os.makedirs(output_raw_segment_class_dir, exist_ok=True)
        os.makedirs(cfg.INTERIM_YOUTUBE_AUDIO_DIR, exist_ok=True)

        video_files = glob.glob(os.path.join(raw_video_dir_for_class, "*.*"))
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        if not video_files: print(f"  Không tìm thấy video nào cho lớp '{class_name}'."); continue

        for video_path in tqdm(video_files, desc=f"Videos for {class_name}"):
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            # print(f"  Đang xử lý video: {video_base_name}") # Bớt print để đỡ rối khi chạy nhiều

            temp_audio_path = os.path.join(cfg.INTERIM_YOUTUBE_AUDIO_DIR, f"prep_fus_{video_base_name}_temp.wav")
            audio_data_full = None
            try:
                with VideoFileClip(video_path) as video_clip:
                    if video_clip.audio is None:
                        print(f"    DEBUG: Video {video_base_name} không có audio. Bỏ qua.")
                        continue
                    video_clip.audio.write_audiofile(temp_audio_path, fps=cfg.AUDIO_SAMPLE_RATE, codec='pcm_s16le', logger=None) # Sửa verbose
                audio_data_full, sr_loaded = librosa.load(temp_audio_path, sr=cfg.AUDIO_SAMPLE_RATE)
                if sr_loaded != cfg.AUDIO_SAMPLE_RATE:
                     print(f"    DEBUG: SR audio loaded {sr_loaded} != target {cfg.AUDIO_SAMPLE_RATE} for {video_base_name}")
            except Exception as e:
                print(f"    Lỗi trích xuất audio từ {video_base_name}: {e}. Bỏ qua.")
                if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
                continue
            finally:
                if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            
            if audio_data_full is None or len(audio_data_full) < int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.1): # Kiểm tra audio có nội dung
                print(f"    DEBUG: Không có dữ liệu audio hoặc quá ngắn sau khi trích xuất từ {video_base_name}. Bỏ qua.")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print(f"    Lỗi mở video {video_base_name} (CV2). Bỏ qua."); continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps == 0: fps = 25
            total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_duration_cv = total_frames_vid / fps if fps > 0 else 0
            num_samples_per_segment_audio = int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)

            max_segments_audio = len(audio_data_full) // num_samples_per_segment_audio
            max_segments_video = int(vid_duration_cv // cfg.AUDIO_SEGMENT_DURATION) if cfg.AUDIO_SEGMENT_DURATION > 0 else 0
            num_segments = min(max_segments_audio, max_segments_video)
            
            print(f"    DEBUG: Video: {video_base_name}, FPS: {fps:.2f}, Total Frames: {total_frames_vid}, CV Duration: {vid_duration_cv:.2f}s")
            print(f"    DEBUG: Audio Total Samples: {len(audio_data_full)}, Samples per Seg: {num_samples_per_segment_audio}")
            print(f"    DEBUG: Max Audio Segs: {max_segments_audio}, Max Video Segs: {max_segments_video}, Num Segments to Process: {num_segments}")


            if num_segments == 0:
                print(f"    DEBUG: Video {video_base_name} quá ngắn hoặc audio/video không khớp. Không tạo segment nào.")
                cap.release(); continue

            for i in range(num_segments):
                segment_tag = f"{video_base_name}_seg{global_segment_counter:05d}"
                # print(f"      Processing segment {i+1}/{num_segments} (ID: {segment_tag})")
                audio_feat_saved, vcam_feat_saved = False, False

                # --- Xử lý và Lưu Audio Segment Gốc và Feature ---
                audio_segment_original_filename = f"{segment_tag}_audio_original.wav"
                audio_segment_original_save_path = os.path.join(output_raw_segment_class_dir, audio_segment_original_filename)
                audio_mfcc_filename = f"{segment_tag}_audio_mfcc.npy"
                audio_mfcc_save_path = os.path.join(output_feature_class_dir, audio_mfcc_filename)
                try:
                    audio_start_s = i * num_samples_per_segment_audio
                    audio_end_s = audio_start_s + num_samples_per_segment_audio
                    audio_segment = audio_data_full[audio_start_s:audio_end_s]

                    if len(audio_segment) < int(num_samples_per_segment_audio * 0.8):
                        print(f"        DEBUG: Audio segment {i} quá ngắn ({len(audio_segment)}). Bỏ qua.")
                        continue
                    
                    sf.write(audio_segment_original_save_path, audio_segment, cfg.AUDIO_SAMPLE_RATE)
                    mfcc_f = extract_mfcc(audio_segment, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
                    mfcc_p, _ = pad_features([mfcc_f], max_len=audio_max_len_for_fusion)
                    if mfcc_p.size == 0:
                        print(f"        DEBUG: MFCC padded rỗng cho audio segment {i}. Bỏ qua.")
                        if os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path)
                        continue
                    mfcc_s, _ = scale_features(mfcc_p, scaler=audio_scaler_for_fusion)
                    np.save(audio_mfcc_save_path, mfcc_s[0])
                    audio_feat_saved = True
                    # print(f"        DEBUG: Audio segment {i} - MFCC saved. Shape: {mfcc_s[0].shape}")
                except Exception as e_a:
                    print(f"        Lỗi xử lý audio segment {i} của {video_base_name}: {e_a}")
                    if os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path) # Dọn dẹp
                    continue

                # --- Xử lý và Lưu VCam Frame Gốc và Feature ---
                vcam_frame_original_filename = f"{segment_tag}_vcam_original.jpg"
                vcam_frame_original_save_path = os.path.join(output_raw_segment_class_dir, vcam_frame_original_filename)
                vcam_features_filename = f"{segment_tag}_vcam_features.npy"
                vcam_features_save_path = os.path.join(output_feature_class_dir, vcam_features_filename)
                frame_bgr_for_vcam = None # Khởi tạo
                try:
                    frame_time_target = (i + 0.5) * cfg.AUDIO_SEGMENT_DURATION
                    frame_num_target = int(frame_time_target * fps)
                    if frame_num_target >= total_frames_vid and total_frames_vid > 0 : frame_num_target = total_frames_vid -1
                    elif total_frames_vid == 0 :
                        print(f"        DEBUG: Video {video_base_name} không có frame nào. Bỏ qua VCam cho segment {i}.")
                        if audio_feat_saved and os.path.exists(audio_mfcc_save_path): os.remove(audio_mfcc_save_path)
                        if audio_feat_saved and os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path)
                        continue

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_target)
                    ret, frame_bgr_for_vcam = cap.read() # Đổi tên biến
                    if not ret or frame_bgr_for_vcam is None:
                        print(f"        DEBUG: Không đọc được frame VCam cho segment {i}. Bỏ qua.")
                        if audio_feat_saved and os.path.exists(audio_mfcc_save_path): os.remove(audio_mfcc_save_path)
                        if audio_feat_saved and os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path)
                        continue
                    cv2.imwrite(vcam_frame_original_save_path, frame_bgr_for_vcam)
                    
                    vcam_input_tensor = preprocess_frame_for_yolo_input(frame_bgr_for_vcam, cfg.VCAM_YOLO_IMG_SIZE)
                    if vcam_input_tensor is None: # Kiểm tra output từ preprocess
                        raise ValueError("Preprocessing VCam frame trả về None.")

                    with torch.no_grad():
                        vcam_feature_vector_tensor = vcam_feature_extractor(vcam_input_tensor.to(device_torch_prep))
                    np.save(vcam_features_save_path, vcam_feature_vector_tensor.cpu().numpy()[0])
                    vcam_feat_saved = True
                    # print(f"        DEBUG: VCam segment {i} - Feature saved. Shape: {vcam_feature_vector_tensor.cpu().numpy()[0].shape}")

                except Exception as e_v:
                    print(f"        Lỗi xử lý VCam segment {i} của {video_base_name}: {e_v}")
                    if audio_feat_saved and os.path.exists(audio_mfcc_save_path): os.remove(audio_mfcc_save_path)
                    if audio_feat_saved and os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path)
                    if os.path.exists(vcam_frame_original_save_path): os.remove(vcam_frame_original_save_path)
                    continue

                metadata_list.append({
                    'segment_id': segment_tag,
                    'original_video': video_base_name,
                    'video_segment_index': i,
                    'audio_mfcc_file': os.path.join(class_name, audio_mfcc_filename),
                    'vcam_features_file': os.path.join(class_name, vcam_features_filename),
                    'audio_segment_original_file': os.path.join(class_name, audio_segment_original_filename),
                    'vcam_frame_original_file': os.path.join(class_name, vcam_frame_original_filename),
                    'label': class_name
                })
                global_segment_counter += 1
            cap.release()

    if metadata_list:
        df_metadata = pd.DataFrame(metadata_list)
        df_metadata.to_csv(cfg.FINETUNE_MULTICLASS_METADATA_FILE, index=False)
        print(f"\nĐã xử lý và lưu {global_segment_counter} cặp đặc trưng.")
        print(f"File metadata cho fusion fine-tune: {cfg.FINETUNE_MULTICLASS_METADATA_FILE}")
    else:
        print("\nKhông có cặp đặc trưng nào được xử lý.")
    print("===== PREPARATION OF MULTI-CLASS FINE-TUNE DATA FINISHED =====")

if __name__ == '__main__':
    required_files_prep_fusion = [
        cfg.AUDIO_SCALER_FOR_FUSION_PREP,
        cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP,
        cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION
    ]
    missing_prep = [f for f in required_files_prep_fusion if not os.path.exists(f)]
    if missing_prep:
        print("LỖI: Thiếu các file pre-trained/helper cần thiết để chuẩn bị dữ liệu fusion:")
        for f_path in missing_prep: print(f" - {f_path}")
    else:
        prepare_fine_tune_data_multiclass()