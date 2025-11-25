# Drone_Detection_Project/src/inference/predict_ircam_yolo_video.py
import os
import glob
import cv2
from ultralytics import YOLO
import time # Để đo thời gian xử lý

# Import config
try:
    from config import project_config as cfg
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from config import project_config as cfg

def predict_on_ir_video(video_path, model, output_video_path=None):
    """
    Thực hiện dự đoán trên video IR, vẽ bounding box và có thể lưu video output.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25 # Mặc định

    writer = None
    if output_video_path:
        # Đảm bảo thư mục output tồn tại
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Hoặc XVID
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Sẽ lưu video kết quả vào: {output_video_path}")


    frame_count = 0
    start_time_total = time.time()

    print(f"Đang xử lý video IR: {os.path.basename(video_path)}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time_frame = time.time()

        # Thực hiện dự đoán bằng YOLO
        # model.predict() nhận frame numpy trực tiếp
        results = model.predict(source=frame, verbose=False, device=model.device) # device từ model đã load

        # results là một list (thường chỉ có 1 phần tử cho 1 ảnh/frame)
        # Mỗi result chứa các box, mask, probs,...
        if results and results[0]:
            result = results[0] # Lấy kết quả cho frame đầu tiên (và duy nhất)
            # Vẽ bounding box lên frame
            annotated_frame = result.plot() # Hàm plot() của YOLO tự vẽ
        else:
            annotated_frame = frame # Giữ frame gốc nếu không có phát hiện

        end_time_frame = time.time()
        processing_time_frame = end_time_frame - start_time_frame
        # print(f"  Frame {frame_count}: Xử lý trong {processing_time_frame:.3f} giây")

        # Hiển thị frame (tùy chọn)
        # cv2.imshow("IR Cam Prediction", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        if writer:
            writer.write(annotated_frame)

    end_time_total = time.time()
    total_processing_time = end_time_total - start_time_total
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    print(f"Xử lý xong video {os.path.basename(video_path)}. Tổng thời gian: {total_processing_time:.2f}s. FPS trung bình: {avg_fps:.2f}")


    cap.release()
    if writer:
        writer.release()
   #  cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg.ensure_output_directories() # Đảm bảo thư mục

    # Tải mô hình IRCam đã huấn luyện
    if not os.path.exists(cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình IRCam tại: {cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH}")
        print("Vui lòng huấn luyện mô hình IRCam trước.")
        exit()

    print(f"Tải mô hình IRCam từ: {cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH}")
    ir_model = YOLO(cfg.IRCAM_YOLO_BEST_MODEL_SAVE_PATH)
    # Không cần ir_model.to(device) vì YOLO tự xử lý device khi load model và predict

    video_files = glob.glob(os.path.join(cfg.IRCAM_VIDEOS_TO_PREDICT_DIR, "*"))
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"Không tìm thấy video nào trong thư mục: {cfg.IRCAM_VIDEOS_TO_PREDICT_DIR}")
    else:
        print(f"\n--- Bắt đầu dự đoán trên các video IR ---")
        for video_path in video_files:
            base_name, ext = os.path.splitext(os.path.basename(video_path))
            output_video_file = os.path.join(cfg.IRCAM_YOLO_REPORTS_FIGURES_DIR, f"{base_name}_predicted{ext}")
            predict_on_ir_video(video_path, ir_model, output_video_path=output_video_file)

    print("--- Dự đoán video IR hoàn tất ---")