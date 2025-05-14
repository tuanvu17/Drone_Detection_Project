# Drone_Detection_Project/src/inference/predict_vcam_yolo_video.py
import os
import glob
import cv2
from ultralytics import YOLO
import time

# Import config
try:
    from config import project_config as cfg
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from config import project_config as cfg

def predict_on_vcam_video(video_path, model, output_video_path=None):
    # Hàm này gần như giống hệt predict_on_ir_video
    # Chỉ khác tên hàm và có thể là một vài tham số nếu cần (nhưng với YOLO thì ít khác)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return
    # ... (phần còn lại giống hệt predict_on_ir_video, chỉ thay tên biến nếu muốn) ...
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25

    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Sẽ lưu video kết quả vào: {output_video_path}")

    frame_count = 0
    start_time_total = time.time()
    print(f"Đang xử lý video VCam: {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        results = model.predict(source=frame, verbose=False, device=model.device)
        if results and results[0]:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        if writer:
            writer.write(annotated_frame)
        # cv2.imshow("VCam Prediction", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
    # ... (phần cuối giống hệt) ...
    end_time_total = time.time()
    total_processing_time = end_time_total - start_time_total
    avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
    print(f"Xử lý xong video {os.path.basename(video_path)}. Tổng thời gian: {total_processing_time:.2f}s. FPS trung bình: {avg_fps:.2f}")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg.ensure_output_directories()

    if not os.path.exists(cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình VCam tại: {cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
        print("Vui lòng huấn luyện mô hình VCam trước.")
        exit()

    print(f"Tải mô hình VCam từ: {cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
    vcam_model = YOLO(cfg.VCAM_YOLO_BEST_MODEL_SAVE_PATH)

    video_files = glob.glob(os.path.join(cfg.VCAM_VIDEOS_TO_PREDICT_DIR, "*"))
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]


    if not video_files:
        print(f"Không tìm thấy video nào trong thư mục: {cfg.VCAM_VIDEOS_TO_PREDICT_DIR}")
    else:
        print(f"\n--- Bắt đầu dự đoán trên các video VCam ---")
        for video_path in video_files:
            base_name, ext = os.path.splitext(os.path.basename(video_path))
            output_video_file = os.path.join(cfg.VCAM_YOLO_REPORTS_FIGURES_DIR, f"{base_name}_predicted{ext}")
            predict_on_vcam_video(video_path, vcam_model, output_video_path=output_video_file)

    print("--- Dự đoán video VCam hoàn tất ---")