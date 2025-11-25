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

# --- Định nghĩa VCam Feature Extractor (PyTorch) ---
class FeatureExtractorVCam(nn.Module):
    def __init__(self, original_yolo_model_sequential, extraction_layer_index):
        super().__init__()
        self.features_extractor_sequential = nn.Sequential(*list(original_yolo_model_sequential.children())[:extraction_layer_index + 1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features_extractor_sequential(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

def preprocess_frame_for_yolo_input(frame_bgr, target_img_size_yolo):
    """
    Tiền xử lý frame BGR từ OpenCV cho đầu vào của mô hình YOLO PyTorch.
    Bao gồm letterboxing, chuyển RGB, chuẩn hóa [0,1], to tensor (B,C,H,W).
    """
    img_h, img_w = frame_bgr.shape[:2]
    new_w, new_h = target_img_size_yolo, target_img_size_yolo # Giả sử target là vuông
    scale = min(new_w / img_w, new_h / img_h)
    resized_w, resized_h = int(round(img_w * scale)), int(round(img_h * scale))

    if img_w != resized_w or img_h != resized_h:
        frame_resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame_bgr.copy()

    dw, dh = (new_w - resized_w) / 2, (new_h - resized_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) # Letterbox padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() # HWC to CHW
    img_tensor /= 255.0  # Chuẩn hóa về [0, 1]
    return img_tensor.unsqueeze(0) # Thêm batch dimension [1, C, H, W]


def prepare_fine_tune_data_multiclass():
    print("===== STARTING PREPARATION OF MULTI-CLASS FINE-TUNE DATA (VCAM + AUDIO) =====")
    if hasattr(cfg, 'ensure_output_directories'):
        cfg.ensure_output_directories()

    print("Đang tải Scaler và MaxLen cho Audio (từ Audio YouTube Fine-tune)...")
    try:
        # Sử dụng scaler và max_len từ quá trình fine-tune audio trên YouTube
        with open(cfg.AUDIO_SCALER_FOR_FUSION_PREP, 'rb') as f:
            audio_scaler_for_fusion = pickle.load(f)
        with open(cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP, 'rb') as f:
            audio_max_len_for_fusion = pickle.load(f)
        print("  Scaler và MaxLen audio cho fusion prep đã tải.")
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy Scaler ({cfg.AUDIO_SCALER_FOR_FUSION_PREP}) hoặc MaxLen ({cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP}). Lỗi: {e}")
        print("Vui lòng chạy pipeline fine-tune audio trên YouTube trước.")
        return
    except Exception as e:
        print(f"Lỗi không xác định khi tải audio scaler/maxlen cho fusion prep: {e}"); return

    print("Đang tải mô hình VCam YOLO để trích xuất đặc trưng (đã fine-tune trên YouTube)...")
    try:
        # Sử dụng mô hình VCam đã fine-tune trên YouTube để trích xuất đặc trưng
        vcam_yolo_for_feature_extraction = YOLO(cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION)
        EXTRACTION_LAYER_IDX = 8
        vcam_feature_extractor = FeatureExtractorVCam(
            vcam_yolo_for_feature_extraction.model.model, # Truy cập nn.Sequential bên trong
            EXTRACTION_LAYER_IDX
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vcam_feature_extractor.to(device).eval()
        print(f"Sử dụng thiết bị: {device} cho VCam feature extraction (từ model: {cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION}).")
    except Exception as e:
        print(f"LỖI: Không thể tải hoặc thiết lập VCam feature extractor: {e}"); return

    metadata_list = []
    global_segment_counter = 0

    # Thư mục gốc để lưu các file segment gốc (cho việc đánh giá sau này)
    # Sẽ tạo các thư mục con theo lớp bên trong
    raw_segments_output_base_dir = os.path.join(cfg.PROJECT_ROOT, 'data', 'processed', 'FineTune_Raw_Segments_For_Eval')
    os.makedirs(raw_segments_output_base_dir, exist_ok=True)


    for class_name in cfg.FUSION_CLASS_NAMES: # Lặp qua các lớp mục tiêu cho fusion
        print(f"\n--- Processing class: {class_name} ---")
        raw_video_dir_for_class = os.path.join(cfg.YOUTUBE_RAW_VIDEOS_BASE_DIR, class_name)
        if not os.path.isdir(raw_video_dir_for_class):
            print(f"  Cảnh báo: Không tìm thấy thư mục video raw cho '{class_name}'. Bỏ qua."); continue

        output_feature_class_dir = os.path.join(cfg.FINETUNE_PAIRED_DATA_BASE_DIR, class_name)
        output_raw_segment_class_dir = os.path.join(raw_segments_output_base_dir, class_name) # Lưu file gốc ở đây
        os.makedirs(output_feature_class_dir, exist_ok=True)
        os.makedirs(output_raw_segment_class_dir, exist_ok=True)
        os.makedirs(cfg.INTERIM_YOUTUBE_AUDIO_DIR, exist_ok=True)

        video_files = glob.glob(os.path.join(raw_video_dir_for_class, "*.*"))
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files: print(f"  Không tìm thấy video cho '{class_name}'."); continue

        for video_path in tqdm(video_files, desc=f"Videos for {class_name}"):
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            # print(f"  Đang xử lý video: {video_base_name}") # Bỏ bớt print để đỡ rối tqdm

            temp_audio_path = os.path.join(cfg.INTERIM_YOUTUBE_AUDIO_DIR, f"{video_base_name}_audio_temp.wav")
            audio_data_full = None
            try:
                with VideoFileClip(video_path) as video_clip:
                    if video_clip.audio is None: print(f"    Video {video_base_name} không có audio. Bỏ qua."); continue
                    video_clip.audio.write_audiofile(temp_audio_path, fps=cfg.AUDIO_SAMPLE_RATE, codec='pcm_s16le', logger=None)
                audio_data_full, sr_loaded = librosa.load(temp_audio_path, sr=cfg.AUDIO_SAMPLE_RATE)
            except Exception as e: print(f"    Lỗi trích xuất audio từ {video_base_name}: {e}. Bỏ qua."); continue
            finally:
                if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
            if audio_data_full is None: continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print(f"    Lỗi mở video {video_base_name}. Bỏ qua."); continue
            fps = cap.get(cv2.CAP_PROP_FPS); total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == 0: fps = 25
            vid_duration_cv = total_frames_vid / fps if fps > 0 else 0
            num_segments = min(len(audio_data_full) // int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE),
                               int(vid_duration_cv // cfg.AUDIO_SEGMENT_DURATION) if cfg.AUDIO_SEGMENT_DURATION > 0 else 0)
            if num_segments == 0: cap.release(); continue

            for i in range(num_segments):
                segment_tag = f"{video_base_name}_seg{global_segment_counter:05d}"
                audio_feat_saved, vcam_feat_saved = False, False # Cờ để kiểm tra

                # --- Xử lý và Lưu Audio Segment Gốc và Feature ---
                audio_segment_original_filename = f"{segment_tag}_audio_original.wav"
                audio_segment_original_save_path = os.path.join(output_raw_segment_class_dir, audio_segment_original_filename)
                audio_mfcc_filename = f"{segment_tag}_audio_mfcc.npy"
                audio_mfcc_save_path = os.path.join(output_feature_class_dir, audio_mfcc_filename)
                try:
                    audio_start_s = int(i * cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
                    audio_end_s = audio_start_s + int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
                    audio_segment = audio_data_full[audio_start_s:audio_end_s]
                    if len(audio_segment) < int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.9): continue
                    sf.write(audio_segment_original_save_path, audio_segment, cfg.AUDIO_SAMPLE_RATE) # Lưu .wav gốc
                    mfcc_f = extract_mfcc(audio_segment, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
                    mfcc_p, _ = pad_features([mfcc_f], max_len=audio_max_len_for_fusion)
                    if mfcc_p.size == 0: continue
                    mfcc_s, _ = scale_features(mfcc_p, scaler=audio_scaler_for_fusion)
                    np.save(audio_mfcc_save_path, mfcc_s[0])
                    audio_feat_saved = True
                except Exception as e_a: print(f"Lỗi audio seg {segment_tag}: {e_a}"); continue

                # --- Xử lý và Lưu VCam Frame Gốc và Feature ---
                vcam_frame_original_filename = f"{segment_tag}_vcam_original.jpg"
                vcam_frame_original_save_path = os.path.join(output_raw_segment_class_dir, vcam_frame_original_filename)
                vcam_features_filename = f"{segment_tag}_vcam_features.npy"
                vcam_features_save_path = os.path.join(output_feature_class_dir, vcam_features_filename)
                try:
                    frame_time_target = (i + 0.5) * cfg.AUDIO_SEGMENT_DURATION
                    frame_num_target = int(frame_time_target * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_target)
                    ret, frame_vcam = cap.read()
                    if not ret or frame_vcam is None:
                        if audio_feat_saved and os.path.exists(audio_mfcc_save_path): os.remove(audio_mfcc_save_path) # Xóa audio feat nếu vcam lỗi
                        if audio_feat_saved and os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path)
                        continue
                    cv2.imwrite(vcam_frame_original_save_path, frame_vcam) # Lưu frame .jpg gốc
                    vcam_input_tensor = preprocess_frame_for_yolo_input(frame_vcam, cfg.VCAM_YOLO_IMG_SIZE) # Dùng VCAM_YOLO_IMG_SIZE
                    with torch.no_grad():
                        vcam_feature_vector = vcam_feature_extractor(vcam_input_tensor.to(device)).cpu().numpy()
                    np.save(vcam_features_save_path, vcam_feature_vector[0])
                    vcam_feat_saved = True
                except Exception as e_v:
                    print(f"Lỗi VCam seg {segment_tag}: {e_v}")
                    if audio_feat_saved and os.path.exists(audio_mfcc_save_path): os.remove(audio_mfcc_save_path)
                    if audio_feat_saved and os.path.exists(audio_segment_original_save_path): os.remove(audio_segment_original_save_path)
                    if os.path.exists(vcam_frame_original_save_path): os.remove(vcam_frame_original_save_path) # Xóa frame nếu có lỗi sau đó
                    continue

                # Ghi Metadata nếu cả hai đều thành công
                metadata_list.append({
                    'segment_id': segment_tag,
                    'original_video': video_base_name,
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
        print(f"\nĐã xử lý {global_segment_counter} đoạn ghép cặp.")
        print(f"Metadata đã lưu vào: {cfg.FINETUNE_MULTICLASS_METADATA_FILE}")
        print(f"Features đã lưu vào: {cfg.FINETUNE_PAIRED_DATA_BASE_DIR}/<LỚP>/")
        print(f"File gốc của segments đã lưu vào: {raw_segments_output_base_dir}/<LỚP>/")

    else:
        print("\nKhông có đoạn ghép cặp nào được xử lý.")
    print("===== PREPARATION OF MULTI-CLASS FINE-TUNE DATA FINISHED =====")

if __name__ == '__main__':
    required_files_prep = [
        cfg.AUDIO_SCALER_FOR_FUSION_PREP, # Scaler từ audio fine-tune YT
        cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP,  # MaxLen từ audio fine-tune YT
        cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION # Model VCam fine-tune YT
    ]
    missing_prep = [f for f in required_files_prep if not os.path.exists(f)]
    if missing_prep:
        print("LỖI: Thiếu các file cần thiết để chuẩn bị dữ liệu fusion:")
        for f_path in missing_prep: print(f" - {f_path}")
    else:
        prepare_fine_tune_data_multiclass()