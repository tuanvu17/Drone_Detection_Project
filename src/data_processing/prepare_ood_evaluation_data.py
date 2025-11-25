# Drone_Detection_Project/src/data_processing/prepare_ood_evaluation_data.py
import os
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import librosa # Mặc dù không dùng trực tiếp ở đây, nhưng để nhất quán
import soundfile as sf # Để lưu audio segment
import pandas as pd
from tqdm import tqdm
import shutil # Để xóa thư mục tạm nếu cần

# Import config
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_prep_ood = os.path.dirname(os.path.abspath(__file__))
    src_dir_prep_ood = os.path.dirname(current_dir_prep_ood)
    project_root_prep_ood = os.path.dirname(src_dir_prep_ood)
    if project_root_prep_ood not in sys.path: sys.path.insert(0, project_root_prep_ood)
    if src_dir_prep_ood not in sys.path: sys.path.insert(0, src_dir_prep_ood)
    from config_loader.loader import get_config

cfg_prep_ood = get_config()

def prepare_ood_data_for_evaluation(
    ood_raw_video_base_dir,       # Đường dẫn đến thư mục video OOD gốc
    ood_processed_segments_base_dir, # Nơi lưu các segment ảnh và audio đã xử lý
    output_metadata_csv_path,     # Đường dẫn file CSV metadata output
    target_segment_duration=1.0,  # Độ dài mỗi segment (giây)
    audio_target_sr=44100,        # Tần số lấy mẫu cho audio
    frame_per_segment=1,          # Số frame ảnh lấy cho mỗi segment (thường là 1, ở giữa)
    image_output_format='.jpg',
    audio_output_format='.wav'
):
    """
    Chuẩn bị dữ liệu từ các video OOD:
    - Phân đoạn video thành các segment 1 giây.
    - Trích xuất frame ảnh đại diện cho mỗi segment.
    - Trích xuất đoạn âm thanh tương ứng cho mỗi segment.
    - Tạo file metadata CSV.
    """
    if not os.path.isdir(ood_raw_video_base_dir):
        print(f"LỖI: Thư mục video OOD gốc không tồn tại: {ood_raw_video_base_dir}")
        return

    # Tạo các thư mục output chính nếu chưa có
    os.makedirs(ood_processed_segments_base_dir, exist_ok=True)
    # Thư mục tạm để lưu audio full trích xuất từ video
    temp_full_audio_dir = os.path.join(ood_processed_segments_base_dir, "temp_full_audio_ood")
    os.makedirs(temp_full_audio_dir, exist_ok=True)

    print(f"--- Bắt đầu chuẩn bị dữ liệu OOD từ: {ood_raw_video_base_dir} ---")
    print(f"Segment đã xử lý sẽ được lưu vào: {ood_processed_segments_base_dir}")
    print(f"File Metadata sẽ được lưu tại: {output_metadata_csv_path}")

    metadata_list = []
    global_segment_id_counter = 0

    # Lấy danh sách các lớp từ thư mục con trong ood_raw_video_base_dir
    # Hoặc sử dụng cfg_prep_ood.MASTER_CLASS_LIST_FUSION nếu bạn muốn chắc chắn về các lớp
    available_classes = [d for d in os.listdir(ood_raw_video_base_dir)
                         if os.path.isdir(os.path.join(ood_raw_video_base_dir, d))]
    
    print(f"Các lớp được tìm thấy trong thư mục OOD video gốc: {available_classes}")


    for class_name in available_classes:
        class_video_source_dir = os.path.join(ood_raw_video_base_dir, class_name)
        
        # Tạo thư mục con cho từng lớp trong thư mục processed segments
        output_class_image_dir = os.path.join(ood_processed_segments_base_dir, class_name, "images")
        output_class_audio_dir = os.path.join(ood_processed_segments_base_dir, class_name, "audio_segments")
        os.makedirs(output_class_image_dir, exist_ok=True)
        os.makedirs(output_class_audio_dir, exist_ok=True)

        print(f"\n--- Đang xử lý lớp: {class_name} ---")
        video_files = [f for f in os.listdir(class_video_source_dir)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]

        if not video_files:
            print(f"  Không tìm thấy file video nào cho lớp '{class_name}'.")
            continue

        for video_filename in tqdm(video_files, desc=f"Processing videos in {class_name}"):
            video_path = os.path.join(class_video_source_dir, video_filename)
            video_basename_no_ext = os.path.splitext(video_filename)[0]

            # 1. Trích xuất toàn bộ audio từ video hiện tại
            temp_extracted_audio_path = os.path.join(temp_full_audio_dir, f"{video_basename_no_ext}_full_audio.wav")
            full_audio_data = None
            video_has_audio = False
            try:
                with VideoFileClip(video_path) as video_clip:
                    if video_clip.audio is not None:
                        video_clip.audio.write_audiofile(temp_extracted_audio_path,
                                                         fps=audio_target_sr,
                                                         codec='pcm_s16le',
                                                         logger=None) # Tắt logger của moviepy
                        full_audio_data, sr_loaded = librosa.load(temp_extracted_audio_path, sr=audio_target_sr)
                        if sr_loaded == audio_target_sr and full_audio_data is not None and len(full_audio_data) > 0:
                            video_has_audio = True
                        else:
                            print(f"    Cảnh báo: Không tải được audio hoặc SR không khớp từ {temp_extracted_audio_path}")
                    else:
                        print(f"    Thông tin: Video {video_filename} không có track âm thanh.")
            except Exception as e_audio_extract:
                print(f"    Lỗi khi trích xuất audio từ {video_filename}: {e_audio_extract}")
            
            # 2. Mở video bằng OpenCV để trích xuất frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    Lỗi: Không thể mở video {video_path} bằng OpenCV.")
                if os.path.exists(temp_extracted_audio_path): os.remove(temp_extracted_audio_path)
                continue
            
            video_cv_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_cv_fps is None or video_cv_fps == 0:
                print(f"    Cảnh báo: Không lấy được FPS từ video {video_filename} qua OpenCV. Sử dụng 25 FPS mặc định.")
                video_cv_fps = 25.0
            
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration_seconds = total_video_frames / video_cv_fps if video_cv_fps > 0 else 0

            # Số lượng segment audio có thể có
            num_audio_samples_per_segment = int(target_segment_duration * audio_target_sr)
            num_possible_audio_segments = 0
            if video_has_audio and full_audio_data is not None:
                num_possible_audio_segments = len(full_audio_data) // num_audio_samples_per_segment
            
            # Số lượng segment video có thể có
            num_possible_video_segments = int(video_duration_seconds // target_segment_duration)

            # Chọn số segment ít hơn để đảm bảo cả audio và video đều có dữ liệu
            num_segments_to_process = min(num_possible_audio_segments if video_has_audio else num_possible_video_segments,
                                          num_possible_video_segments)
            
            if num_segments_to_process == 0:
                print(f"    Video {video_filename} quá ngắn hoặc không có audio/video hợp lệ để tạo segment 1s.")
                cap.release()
                if os.path.exists(temp_extracted_audio_path): os.remove(temp_extracted_audio_path)
                continue

            # print(f"    Video: {video_filename}, FPS: {video_cv_fps:.2f}, Duration: {video_duration_seconds:.2f}s, Segments: {num_segments_to_process}")

            for i in range(num_segments_to_process):
                segment_start_time_sec = i * target_segment_duration
                segment_end_time_sec = (i + 1) * target_segment_duration
                
                segment_id_str = f"{video_basename_no_ext}_seg{global_segment_id_counter:06d}"

                # --- Xử lý và Lưu Frame Ảnh ---
                frame_to_save_path_relative = ""
                try:
                    # Lấy frame ở giữa segment
                    target_frame_number = int((segment_start_time_sec + target_segment_duration / 2) * video_cv_fps)
                    if target_frame_number >= total_video_frames and total_video_frames > 0:
                        target_frame_number = total_video_frames - 1 # Lấy frame cuối nếu vượt quá
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
                    ret_frame, frame = cap.read()
                    if ret_frame and frame is not None:
                        image_filename = f"{segment_id_str}_vcam{image_output_format}"
                        image_save_abs_path = os.path.join(output_class_image_dir, image_filename)
                        cv2.imwrite(image_save_abs_path, frame)
                        # Đường dẫn tương đối từ ood_processed_segments_base_dir
                        frame_to_save_path_relative = os.path.join(class_name, "images", image_filename)
                    else:
                        print(f"      Cảnh báo: Không thể đọc frame {target_frame_number} cho segment {i} của video {video_filename}")
                        continue # Bỏ qua segment này nếu không có frame
                except Exception as e_frame:
                    print(f"      Lỗi khi xử lý frame cho segment {i} của video {video_filename}: {e_frame}")
                    continue

                # --- Xử lý và Lưu Đoạn Âm thanh (nếu video có audio) ---
                audio_to_save_path_relative = ""
                if video_has_audio and full_audio_data is not None:
                    try:
                        audio_start_sample = int(segment_start_time_sec * audio_target_sr)
                        audio_end_sample = int(segment_end_time_sec * audio_target_sr)
                        audio_segment_data = full_audio_data[audio_start_sample:audio_end_sample]

                        if len(audio_segment_data) >= int(num_audio_samples_per_segment * 0.5): # Ngưỡng tối thiểu
                            audio_filename = f"{segment_id_str}_audio{audio_output_format}"
                            audio_save_abs_path = os.path.join(output_class_audio_dir, audio_filename)
                            sf.write(audio_save_abs_path, audio_segment_data, audio_target_sr)
                            audio_to_save_path_relative = os.path.join(class_name, "audio_segments", audio_filename)
                        else:
                            # print(f"      Cảnh báo: Đoạn audio cho segment {i} của video {video_filename} quá ngắn. Không lưu audio.")
                            # Nếu không có audio, bạn có thể quyết định bỏ qua cả segment hoặc vẫn giữ lại với audio rỗng
                            # Hiện tại, nếu audio không đạt, ta vẫn giữ frame ảnh (nếu có)
                            pass # audio_to_save_path_relative sẽ rỗng
                    except Exception as e_audio_seg:
                        print(f"      Lỗi khi xử lý đoạn audio cho segment {i} của video {video_filename}: {e_audio_seg}")
                
                # Chỉ thêm vào metadata nếu có frame ảnh
                if frame_to_save_path_relative:
                    metadata_list.append({
                        'segment_id': segment_id_str,
                        'original_video_filename': video_filename,
                        'video_class_label': class_name, # Lớp của video gốc
                        'segment_index_in_video': i,
                        'path_to_vcam_frame': frame_to_save_path_relative,
                        'path_to_audio_segment': audio_to_save_path_relative if audio_to_save_path_relative else pd.NA, # pd.NA cho giá trị thiếu
                        'ground_truth_label': class_name # Nhãn ground truth cho cả segment này
                    })
                    global_segment_id_counter += 1
            
            cap.release()
            if os.path.exists(temp_extracted_audio_path): # Xóa file audio tạm sau khi xử lý xong video
                try: os.remove(temp_extracted_audio_path)
                except: pass


    # Dọn dẹp thư mục audio tạm cuối cùng
    if os.path.exists(temp_full_audio_dir):
        try:
            shutil.rmtree(temp_full_audio_dir)
            print(f"Đã xóa thư mục audio tạm: {temp_full_audio_dir}")
        except Exception as e_rm_temp:
            print(f"Lỗi khi xóa thư mục audio tạm: {e_rm_temp}")


    if metadata_list:
        df_metadata = pd.DataFrame(metadata_list)
        df_metadata.to_csv(output_metadata_csv_path, index=False)
        print(f"\nĐã xử lý và lưu {global_segment_id_counter} segment OOD.")
        print(f"File Metadata OOD đã được lưu tại: {output_metadata_csv_path}")
    else:
        print("\nKhông có segment OOD nào được xử lý và lưu.")

    print("===== CHUẨN BỊ DỮ LIỆU OOD HOÀN TẤT =====")


if __name__ == '__main__':
    # Đảm bảo các thư mục output được tạo
    if hasattr(cfg_prep_ood, 'ensure_output_directories') and callable(cfg_prep_ood.ensure_output_directories):
        cfg_prep_ood.ensure_output_directories()
    else: # Tự tạo
        os.makedirs(cfg_prep_ood.EVALUATION_TEST_SEGMENTS_BASE_DIR, exist_ok=True)
        # Tạo thư mục con cho từng lớp nếu cfg.MASTER_CLASS_LIST_FUSION được định nghĩa
        if hasattr(cfg_prep_ood, 'MASTER_CLASS_LIST_FUSION') and cfg_prep_ood.MASTER_CLASS_LIST_FUSION:
            for class_n_main in cfg_prep_ood.MASTER_CLASS_LIST_FUSION:
                os.makedirs(os.path.join(cfg_prep_ood.EVALUATION_TEST_SEGMENTS_BASE_DIR, class_n_main, "images"), exist_ok=True)
                os.makedirs(os.path.join(cfg_prep_ood.EVALUATION_TEST_SEGMENTS_BASE_DIR, class_n_main, "audio_segments"), exist_ok=True)
        else: # Nếu không, sẽ tạo khi duyệt video
            print("Cảnh báo: MASTER_CLASS_LIST_FUSION không được định nghĩa trong config, thư mục con theo lớp sẽ được tạo khi xử lý video.")


    prepare_ood_data_for_evaluation(
        ood_raw_video_base_dir=cfg_prep_ood.EVALUATION_TEST_RAW_VIDEO_DIR,
        ood_processed_segments_base_dir=cfg_prep_ood.EVALUATION_TEST_SEGMENTS_BASE_DIR, # Sửa lại cho đúng config của bạn
        output_metadata_csv_path=cfg_prep_ood.EVALUATION_TEST_GROUND_TRUTH_METADATA_FILE, # Sửa lại cho đúng config
        target_segment_duration=1.0,
        audio_target_sr=cfg_prep_ood.AUDIO_SAMPLE_RATE,
        frame_per_segment=1,
        image_output_format='.jpg',
        audio_output_format='.wav'
    )