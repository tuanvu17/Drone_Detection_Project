# Drone_Detection_Project/src/inference/predict_vcam_youtube_finetuned_video.py
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
    # Giả sử file này nằm trong src/inference/
    PROJECT_ROOT_INF_VCAM_YT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT_INF_VCAM_YT not in sys.path:
        sys.path.append(PROJECT_ROOT_INF_VCAM_YT)
    from config import project_config as cfg

def predict_on_vcam_youtube_video(video_path, model, output_video_path=None, imgsz=640, conf_thresh=0.25):
    """
    Thực hiện dự đoán trên video sử dụng mô hình VCam YOLO đã fine-tune trên YouTube data.
    Vẽ bounding box và có thể lưu video output.

    Args:
        video_path (str): Đường dẫn đến video cần dự đoán.
        model (YOLO): Đối tượng mô hình YOLO đã tải.
        output_video_path (str, optional): Đường dẫn để lưu video kết quả. Nếu None, không lưu.
        imgsz (int): Kích thước ảnh đầu vào cho mô hình YOLO.
        conf_thresh (float): Ngưỡng tin cậy để hiển thị bounding box.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Lấy thông tin video gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        print(f"Cảnh báo: Không lấy được FPS cho video {os.path.basename(video_path)}. Sử dụng mặc định 25 FPS.")
        fps = 25

    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec cho MP4
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Video kết quả sẽ được lưu tại: {output_video_path}")

    frame_count = 0
    total_inference_time = 0
    print(f"\nĐang xử lý video: {os.path.basename(video_path)} bằng mô hình VCam YouTube Fine-tuned")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time_frame = time.time()

        # Thực hiện dự đoán bằng mô hình YOLO đã tải
        # model.predict() có thể nhận trực tiếp frame numpy
        results = model.predict(source=frame, imgsz=imgsz, conf=conf_thresh, verbose=False, device=model.device)

        end_time_frame = time.time()
        total_inference_time += (end_time_frame - start_time_frame)

        # results là một list, thường chỉ có 1 phần tử cho 1 ảnh/frame
        annotated_frame = frame # Mặc định là frame gốc nếu không có phát hiện
        if results and results[0] and results[0].boxes: # Kiểm tra có box nào không
            # Sử dụng hàm plot() của YOLO để vẽ bounding box, nhãn và confidence
            annotated_frame = results[0].plot()
        else:
            # Nếu không có phát hiện, vẫn giữ frame gốc
            pass # annotated_frame đã là frame gốc

        # Hiển thị frame (tùy chọn, bỏ comment nếu muốn xem trực tiếp)
        # cv2.imshow(f"VCam YouTube Fine-tuned Prediction - {os.path.basename(video_path)}", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        if writer:
            writer.write(annotated_frame)

    # Giải phóng tài nguyên
    cap.release()
    if writer:
        writer.release()
        print(f"  Đã lưu video kết quả: {os.path.basename(output_video_path)}")
    # cv2.destroyAllWindows() # Đóng tất cả cửa sổ OpenCV nếu có imshow

    avg_inference_time_per_frame = total_inference_time / frame_count if frame_count > 0 else 0
    print(f"Xử lý xong video {os.path.basename(video_path)}.")
    print(f"  Tổng số frame: {frame_count}")
    print(f"  Tổng thời gian dự đoán: {total_inference_time:.2f} giây")
    print(f"  Thời gian dự đoán trung bình/frame: {avg_inference_time_per_frame*1000:.2f} ms")
    print(f"  FPS dự đoán trung bình: {1/avg_inference_time_per_frame if avg_inference_time_per_frame > 0 else 0:.2f}")


if __name__ == "__main__":
    # Đảm bảo các thư mục output tồn tại
    if hasattr(cfg, 'ensure_output_directories'):
        cfg.ensure_output_directories()
    else: # Tạo thủ công nếu hàm không có trong config hoặc không được gọi
        os.makedirs(cfg.VCAM_YOLO_REPORTS_FIGURES_DIR, exist_ok=True) # Cần thư mục này để lưu video output
        # Thêm các thư mục khác nếu script này cần đến

    # Kiểm tra sự tồn tại của mô hình VCam đã fine-tune trên YouTube
    if not os.path.exists(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH):
        print(f"LỖI: Không tìm thấy file mô hình VCam đã fine-tune trên YouTube tại: {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH}")
        print("Vui lòng chạy pipeline fine-tuning VCam trên dữ liệu YouTube trước ('src.main_finetune_vcam_youtube.py').")
        exit()

    print(f"Tải mô hình VCam đã fine-tune trên YouTube từ: {cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH}")
    try:
        # Tải mô hình YOLO
        vcam_youtube_model = YOLO(cfg.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH)
        # Không cần .to(device) rõ ràng ở đây, YOLO sẽ tự xử lý hoặc dùng device đã lưu trong model
        print(f"  Mô hình VCam YouTube Fine-tuned được tải thành công. Chạy trên thiết bị: {vcam_youtube_model.device}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình VCam YouTube Fine-tuned: {e}")
        exit()


    # Đường dẫn đến thư mục chứa video cần dự đoán
    videos_to_predict_input_dir = os.path.join(cfg.YOUTUBE_RAW_VIDEOS_BASE_DIR, 'VIDEO2PREDICT')
    # Cần tạo thư mục này nếu chưa có và đặt video vào
    os.makedirs(videos_to_predict_input_dir, exist_ok=True)
    print(f"\nSẽ tìm video để dự đoán trong thư mục: {videos_to_predict_input_dir}")


    video_files = glob.glob(os.path.join(videos_to_predict_input_dir, "*.*"))
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"Không tìm thấy video nào trong thư mục: {videos_to_predict_input_dir}")
        print(f"Vui lòng đặt các video bạn muốn dự đoán vào thư mục này.")
    else:
        print(f"\n--- Bắt đầu dự đoán trên {len(video_files)} video bằng mô hình VCam YouTube Fine-tuned ---")

        # Thư mục để lưu video kết quả (sử dụng lại cấu hình của VCam gốc hoặc tạo mới)
        output_video_dir = cfg.VCAM_YOLO_REPORTS_FIGURES_DIR # Hoặc cfg.FUSION_REPORTS_FIGURES_DIR nếu muốn gom chung
        os.makedirs(output_video_dir, exist_ok=True)


        for video_file_path in video_files:
            base_name, ext = os.path.splitext(os.path.basename(video_file_path))
            # Tạo tên file output rõ ràng hơn
            output_file_path = os.path.join(output_video_dir, f"{base_name}_vcam_youtube_predicted{ext}")

            predict_on_vcam_youtube_video(
                video_path=video_file_path,
                model=vcam_youtube_model,
                output_video_path=output_file_path,
                imgsz=cfg.VCAM_YOUTUBE_FINETUNE_IMG_SIZE, # Sử dụng imgsz đã dùng khi fine-tune
                conf_thresh=0.3 # Ngưỡng tin cậy, bạn có thể điều chỉnh
            )

    print("\n--- Dự đoán video bằng mô hình VCam YouTube Fine-tuned hoàn tất ---")