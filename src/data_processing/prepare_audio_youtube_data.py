# Drone_Detection_Project/src/data_processing/prepare_audio_youtube_data.py
import os
import glob
import librosa
import numpy as np
import soundfile as sf # Cần cài đặt: pip install soundfile
from moviepy.video.io.VideoFileClip import VideoFileClip # Sử dụng import đã sửa
import pandas as pd
from tqdm import tqdm

try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_prep_audio_yt = os.path.dirname(os.path.abspath(__file__))
    src_dir_prep_audio_yt = os.path.dirname(current_dir_prep_audio_yt)
    project_root_prep_audio_yt = os.path.dirname(src_dir_prep_audio_yt)
    if project_root_prep_audio_yt not in sys.path: sys.path.insert(0, project_root_prep_audio_yt)
    if src_dir_prep_audio_yt not in sys.path: sys.path.insert(0, src_dir_prep_audio_yt)
    from config_loader.loader import get_config

cfg = get_config()

def prepare_audio_segments_from_youtube_videos():
    print("===== STARTING PREPARATION OF AUDIO SEGMENTS FROM YOUTUBE VIDEOS =====")
    if hasattr(cfg, 'ensure_output_directories'):
        cfg.ensure_output_directories() # Đảm bảo các thư mục output tồn tại

    metadata_list_audio_yt = []
    global_audio_segment_counter = 0

    # Lặp qua các lớp được định nghĩa cho fine-tune audio (nên là MASTER_CLASS_LIST_FUSION)
    for class_name in cfg.AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE:
        print(f"\n--- Processing YouTube videos for class: {class_name} ---")
        
        # Đường dẫn đến thư mục video raw cho lớp hiện tại
        raw_video_class_dir = os.path.join(cfg.YOUTUBE_RAW_VIDEOS_BASE_DIR, class_name)
        if not os.path.isdir(raw_video_class_dir):
            print(f"  Cảnh báo: Không tìm thấy thư mục video raw cho lớp '{class_name}' tại '{raw_video_class_dir}'. Bỏ qua.")
            continue

        # Thư mục output để lưu các segment audio .wav cho lớp này
        output_audio_segment_class_dir = os.path.join(cfg.AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR, class_name)
        os.makedirs(output_audio_segment_class_dir, exist_ok=True)
        os.makedirs(cfg.INTERIM_YOUTUBE_AUDIO_DIR, exist_ok=True) # Thư mục tạm

        video_files = glob.glob(os.path.join(raw_video_class_dir, "*.*"))
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print(f"  Không tìm thấy video nào cho lớp '{class_name}' trong: {raw_video_class_dir}")
            continue

        for video_path in tqdm(video_files, desc=f"Videos for {class_name}"):
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"  Đang trích xuất audio segments từ video: {video_base_name}")

            temp_extracted_audio_path = os.path.join(cfg.INTERIM_YOUTUBE_AUDIO_DIR, f"{video_base_name}_full_audio_temp.wav")
            audio_data_full = None
            try:
                with VideoFileClip(video_path) as video_clip:
                    if video_clip.audio is None:
                        print(f"    Video {video_base_name} không có track audio. Bỏ qua.")
                        continue
                    # Ghi audio vào file tạm với đúng sample rate
                    video_clip.audio.write_audiofile(temp_extracted_audio_path, 
                                                     fps=cfg.AUDIO_SAMPLE_RATE, 
                                                     codec='pcm_s16le', # Codec WAV chuẩn
                                                     logger=None) 
                audio_data_full, sr_loaded = librosa.load(temp_extracted_audio_path, sr=cfg.AUDIO_SAMPLE_RATE)
                if sr_loaded != cfg.AUDIO_SAMPLE_RATE:
                     print(f"    CẢNH BÁO: Audio được load với SR {sr_loaded} thay vì {cfg.AUDIO_SAMPLE_RATE} từ file tạm.")
            except Exception as e:
                print(f"    Lỗi trích xuất audio đầy đủ từ {video_base_name}: {e}. Bỏ qua video này.")
                if os.path.exists(temp_extracted_audio_path): os.remove(temp_extracted_audio_path)
                continue
            finally:
                if os.path.exists(temp_extracted_audio_path): os.remove(temp_extracted_audio_path)
            
            if audio_data_full is None or len(audio_data_full) == 0:
                print(f"    Không có dữ liệu audio sau khi trích xuất từ {video_base_name}. Bỏ qua.")
                continue

            # Phân đoạn audio_data_full thành các segment 1 giây
            num_samples_per_segment = int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE)
            num_segments_in_video = len(audio_data_full) // num_samples_per_segment

            if num_segments_in_video == 0:
                print(f"    Audio từ {video_base_name} quá ngắn để tạo segment 1 giây. Bỏ qua.")
                continue

            for i in range(num_segments_in_video):
                audio_segment_start = i * num_samples_per_segment
                audio_segment_end = audio_segment_start + num_samples_per_segment
                audio_segment_data = audio_data_full[audio_segment_start:audio_segment_end]

                if len(audio_segment_data) < num_samples_per_segment * 0.9: # Bỏ qua nếu segment quá ngắn
                    continue

                segment_filename = f"{video_base_name}_seg{global_audio_segment_counter:06d}.wav"
                segment_save_path = os.path.join(output_audio_segment_class_dir, segment_filename)

                try:
                    sf.write(segment_save_path, audio_segment_data, cfg.AUDIO_SAMPLE_RATE)
                    
                    metadata_list_audio_yt.append({
                        'segment_id': f"audio_yt_seg{global_audio_segment_counter:06d}",
                        'original_video': video_base_name,
                        'audio_segment_file': os.path.join(class_name, segment_filename), # Đường dẫn tương đối
                        'label': class_name
                    })
                    global_audio_segment_counter += 1
                except Exception as e_write:
                    print(f"    Lỗi khi lưu audio segment {segment_filename}: {e_write}")
        
    # Lưu metadata
    if metadata_list_audio_yt:
        df_metadata_audio_yt = pd.DataFrame(metadata_list_audio_yt)
        df_metadata_audio_yt.to_csv(cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE, index=False)
        print(f"\nĐã xử lý và lưu {global_audio_segment_counter} audio segments cho fine-tuning.")
        print(f"Metadata đã được lưu vào: {cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE}")
    else:
        print("\nKhông có audio segment nào được xử lý cho fine-tuning.")

    print("===== PREPARATION OF AUDIO SEGMENTS FROM YOUTUBE VIDEOS FINISHED =====")

if __name__ == '__main__':
    prepare_audio_segments_from_youtube_videos()