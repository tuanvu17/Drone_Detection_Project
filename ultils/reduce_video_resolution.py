import os
import cv2
import subprocess
from pathlib import Path
import logging
import tempfile

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_video_quality_opencv(input_path, output_path, target_height=480):
    """
    Giảm chất lượng video xuống 480p sử dụng OpenCV + FFmpeg (chỉ cho âm thanh)
    
    Args:
        input_path: Đường dẫn video đầu vào
        output_path: Đường dẫn video đầu ra
        target_height: Chiều cao mục tiêu (mặc định 480p)
    """
    
    try:
        # Tạo file tạm cho video không có âm thanh
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Bước 1: Xử lý video (không có âm thanh) bằng OpenCV
        logging.info(f"🎬 Bước 1/2: Xử lý video...")
        
        # Mở video đầu vào
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logging.error(f"❌ Không thể mở video: {input_path}")
            return False
        
        # Lấy thông tin video gốc
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tính toán kích thước mới (giữ tỷ lệ khung hình)
        if original_height <= target_height:
            logging.info(f"⏭️ Video đã có độ phân giải <= {target_height}p")
            # Nếu video đã nhỏ hơn hoặc bằng 480p, chỉ cần copy
            cap.release()
            os.unlink(temp_video_path)
            
            # Copy trực tiếp file gốc
            import shutil
            shutil.copy2(input_path, output_path)
            logging.info(f"✅ Đã copy: {os.path.basename(output_path)}")
            return True
            
        aspect_ratio = original_width / original_height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        
        # Đảm bảo width là số chẵn (yêu cầu của một số codec)
        if new_width % 2 != 0:
            new_width += 1
        
        logging.info(f"📏 Resize: {original_width}x{original_height} → {new_width}x{new_height}")
        
        # Cấu hình video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_width, new_height))
        
        if not out.isOpened():
            logging.error(f"❌ Không thể tạo file tạm: {temp_video_path}")
            cap.release()
            os.unlink(temp_video_path)
            return False
        
        # Xử lý từng frame
        frame_count = 0
        last_progress = -1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Ghi frame
            out.write(resized_frame)
            
            frame_count += 1
            
            # Hiển thị tiến độ
            progress = int((frame_count / total_frames) * 100)
            if progress != last_progress and progress % 10 == 0:
                logging.info(f"🎬 Xử lý video: {progress}% ({frame_count}/{total_frames} frames)")
                last_progress = progress
        
        # Đóng video
        cap.release()
        out.release()
        
        # Bước 2: Ghép âm thanh từ video gốc bằng FFmpeg
        logging.info(f"🔊 Bước 2/2: Ghép âm thanh...")
        
        # Kiểm tra xem có FFmpeg không
        if check_ffmpeg_silent():
            # Sử dụng FFmpeg để ghép âm thanh
            cmd = [
                'ffmpeg',
                '-i', temp_video_path,      # Video đã resize (không có âm thanh)
                '-i', input_path,           # Video gốc (để lấy âm thanh)
                '-c:v', 'copy',             # Copy video stream
                '-c:a', 'aac',              # Encode âm thanh thành AAC
                '-b:a', '128k',             # Bitrate âm thanh 128kbps
                '-shortest',                # Dừng khi stream ngắn nhất kết thúc
                '-y',                       # Ghi đè file output
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"✅ Hoàn thành: {os.path.basename(output_path)}")
                success = True
            else:
                logging.error(f"❌ Lỗi khi ghép âm thanh: {result.stderr}")
                # Fallback: copy file video không có âm thanh
                import shutil
                shutil.move(temp_video_path, output_path)
                logging.warning(f"⚠️ Đã lưu video không có âm thanh: {os.path.basename(output_path)}")
                success = True
        else:
            # Không có FFmpeg, chỉ lưu video không có âm thanh
            import shutil
            shutil.move(temp_video_path, output_path)
            logging.warning(f"⚠️ Không có FFmpeg - Video không có âm thanh: {os.path.basename(output_path)}")
            success = True
        
        # Xóa file tạm
        try:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        except:
            pass
        
        return success
        
    except Exception as e:
        logging.error(f"❌ Lỗi khi xử lý {input_path}: {e}")
        # Xóa file tạm nếu có lỗi
        try:
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        except:
            pass
        return False

def process_videos_opencv(input_dir, output_dir=None):
    """
    Xử lý tất cả video trong thư mục sử dụng OpenCV
    """
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logging.error(f"❌ Thư mục không tồn tại: {input_dir}")
        return
    
    # Tạo thư mục output
    if output_dir is None:
        output_path = input_path.parent / f"{input_path.name}_480p"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(exist_ok=True)
    logging.info(f"📁 Thư mục output: {output_path}")
    
    # Các định dạng video hỗ trợ
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}
    
    # Tìm tất cả file video
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not video_files:
        logging.warning("⚠️ Không tìm thấy file video nào trong thư mục!")
        return
    
    logging.info(f"🎥 Tìm thấy {len(video_files)} file video")
    
    # Xử lý từng video
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        logging.info(f"📊 Video {i}/{len(video_files)}: {video_file.name}")
        
        # Tạo tên file output
        output_file = output_path / f"{video_file.stem}_480p.mp4"
        
        # Kiểm tra nếu file output đã tồn tại
        if output_file.exists():
            logging.info(f"⏭️ Bỏ qua (đã tồn tại): {output_file.name}")
            continue
        
        # Xử lý video
        if reduce_video_quality_opencv(str(video_file), str(output_file)):
            successful += 1
        else:
            failed += 1
    
    # Thống kê kết quả
    logging.info(f"\n📈 KẾT QUẢ CUỐI CÙNG:")
    logging.info(f"✅ Thành công: {successful} video")
    logging.info(f"❌ Thất bại: {failed} video")
    logging.info(f"📁 Thư mục output: {output_path}")

def check_opencv():
    """Kiểm tra xem OpenCV đã được cài đặt chưa"""
    try:
        import cv2
        logging.info(f"✅ OpenCV đã được cài đặt - Phiên bản: {cv2.__version__}")
        return True
    except ImportError:
        logging.error("❌ OpenCV chưa được cài đặt")
        logging.error("💡 Cài đặt bằng lệnh: pip install opencv-python")
        return False

def check_ffmpeg_silent():
    """Kiểm tra FFmpeg một cách im lặng (không log)"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True)
        return True
    except FileNotFoundError:
        return False

def check_ffmpeg():
    """Kiểm tra FFmpeg và hiển thị thông báo"""
    if check_ffmpeg_silent():
        logging.info("✅ FFmpeg có sẵn - Video sẽ có âm thanh")
        return True
    else:
        logging.warning("⚠️ FFmpeg không có - Video sẽ không có âm thanh")
        logging.info("💡 Để có âm thanh, cài đặt FFmpeg:")
        logging.info("   - Windows: winget install ffmpeg")
        logging.info("   - Hoặc tải từ: https://ffmpeg.org/download.html")
        return False

if __name__ == "__main__":
    # Đường dẫn thư mục video của bạn
    VIDEO_DIRECTORY = r"/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/raw/YouTube_Videos_Raw/HELICOPTER"
    
    print("🎬 CÔNG CỤ GIẢM CHẤT LƯỢNG VIDEO XUỐNG 480P (CÓ ÂM THANH)")
    print("=" * 60)
    
    # Kiểm tra OpenCV
    if not check_opencv():
        print("\n💡 Để cài đặt OpenCV, chạy lệnh:")
        print("pip install opencv-python")
        exit(1)
    
    # Kiểm tra FFmpeg (không bắt buộc)
    has_ffmpeg = check_ffmpeg()
    
    if not has_ffmpeg:
        print("\n⚠️  CẢNH BÁO: Video output sẽ KHÔNG CÓ ÂM THANH!")
        choice = input("Bạn có muốn tiếp tục không? (y/n): ").lower().strip()
        if choice not in ['y', 'yes', 'có']:
            print("Đã hủy. Vui lòng cài đặt FFmpeg trước.")
            exit(1)
    
    # Xử lý video
    process_videos_opencv(VIDEO_DIRECTORY)
    
    print("\n🎉 HOÀN THÀNH!")
    if has_ffmpeg:
        print("✅ Video có cả hình ảnh và âm thanh")
    else:
        print("⚠️ Video chỉ có hình ảnh (không có âm thanh)")
    input("Nhấn Enter để thoát...")