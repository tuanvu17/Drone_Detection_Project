import os
import subprocess
import glob

# Đường dẫn thư mục chứa video DRONE
# input_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/raw/YouTube_Videos_Raw/DRONE"
# output_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/processed/DRONE_segments"

# # Đường dẫn thư mục chứa video HELICOPER
# input_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/raw/YouTube_Videos_Raw/HELICOPTER"
# output_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/processed/HELICOPTER"


# Đường dẫn thư mục chứa video HELICOPER
input_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/raw/YouTube_Videos_Raw/BACKGROUND"
output_dir = "/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/data/processed/BACKGROUND"


# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Định dạng video cần xử lý (có thể mở rộng: mp4, avi, mov, ...)
video_extensions = ["*.mp4", "*.avi", "*.mov"]

# Lặp qua tất cả các video trong thư mục
for ext in video_extensions:
    for video_path in glob.glob(os.path.join(input_dir, ext)):
        video_name = os.path.basename(video_path).split('.')[0]
        output_pattern = os.path.join(output_dir, f"{video_name}_%03d.mp4")
        
        # Lệnh FFmpeg để chia video
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", "10",
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]
        
        try:
            # Chạy lệnh FFmpeg
            subprocess.run(cmd, check=True)
            print(f"Đã xử lý: {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi xử lý {video_path}: {e}")

print("Hoàn tất việc chia video!")