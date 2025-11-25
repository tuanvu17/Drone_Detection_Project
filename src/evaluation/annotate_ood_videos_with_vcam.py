# Drone_Detection_Project/src/evaluation/annotate_ood_videos_with_vcam.py
import os
import cv2
from ultralytics import YOLO
import torch
from tqdm import tqdm

# Import config
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_annotate = os.path.dirname(os.path.abspath(__file__))
    src_dir_annotate = os.path.dirname(current_dir_annotate)
    project_root_annotate = os.path.dirname(src_dir_annotate)
    if project_root_annotate not in sys.path: sys.path.insert(0, project_root_annotate)
    if src_dir_annotate not in sys.path: sys.path.insert(0, src_dir_annotate)
    from config_loader.loader import get_config

cfg_annotate = get_config()

def annotate_videos_from_ood_set(
    ood_video_base_dir,
    output_annotated_base_dir,
    vcam_model_path,
    img_size,
    conf_thresh_viz=0.4,
    frame_skip=15 # Xử lý 1 frame mỗi N frames để giảm số lượng ảnh, ví dụ mỗi 0.5 giây nếu video 30fps
):
    """
    Đọc các video từ bộ OOD, chạy VCam YOLO để phát hiện và vẽ bounding box,
    sau đó lưu các frame đã annotate.
    """
    if not os.path.exists(ood_video_base_dir):
        print(f"LỖI: Thư mục video OOD không tồn tại: {ood_video_base_dir}")
        return

    if not os.path.exists(vcam_model_path):
        print(f"LỖI: Không tìm thấy mô hình VCam tại: {vcam_model_path}")
        return

    print(f"Đang tải mô hình VCam từ: {vcam_model_path}")
    try:
        # SỬA ĐOẠN NÀY
        if torch.cuda.is_available():
            device = 'cuda:0' # Sử dụng định dạng 'cuda:0' cho GPU đầu tiên
        else:
            device = 'cpu'
        model = YOLO(vcam_model_path)
        model.to(device) # Chuyển model sang device
        print(f"Mô hình VCam đã tải và chuyển sang device: {device}")
    except Exception as e:
        print(f"Lỗi khi tải hoặc chuyển mô hình VCam: {e}")
        return

    # Lấy tên các lớp từ model YOLO
    if isinstance(model.names, dict):
        class_names_from_model = [model.names[i] for i in sorted(model.names.keys())]
    elif isinstance(model.names, list):
        class_names_from_model = model.names
    else:
        print("LỖI: Không thể xác định class_names_from_model từ model.names.")
        return

    # Duyệt qua các thư mục lớp trong ood_video_base_dir
    for class_name in os.listdir(ood_video_base_dir):
        class_video_dir = os.path.join(ood_video_base_dir, class_name)
        if not os.path.isdir(class_video_dir):
            continue

        output_class_annotated_dir = os.path.join(output_annotated_base_dir, class_name)
        os.makedirs(output_class_annotated_dir, exist_ok=True)
        print(f"\nĐang xử lý lớp: {class_name}")

        video_files = [f for f in os.listdir(class_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        for video_filename in tqdm(video_files, desc=f"Annotating videos in {class_name}"):
            video_path = os.path.join(class_video_dir, video_filename)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cảnh báo: Không thể mở video {video_path}")
                continue

            frame_count = 0
            processed_frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_skip != 0: # Bỏ qua frame để xử lý nhanh hơn
                    continue

                # Chạy YOLO predict trên frame hiện tại
                results = model.predict(source=frame, imgsz=img_size, conf=0.25, verbose=False, device=device) # conf thấp hơn để bắt được nhiều đối tượng tiềm năng

                drawn_frame = frame.copy()
                found_detection_to_draw = False

                if results and results[0].boxes and hasattr(results[0].boxes, 'conf') and results[0].boxes.conf is not None and len(results[0].boxes.conf) > 0:
                    for box_idx in range(len(results[0].boxes.conf)):
                        conf = float(results[0].boxes.conf[box_idx])
                        cls_id = int(results[0].boxes.cls[box_idx])

                        if cls_id >= len(class_names_from_model):
                            continue # Bỏ qua nếu ID lớp không hợp lệ

                        if conf >= conf_thresh_viz:
                            found_detection_to_draw = True
                            coords = results[0].boxes.xyxy[box_idx].cpu().numpy().astype(int)
                            pred_class_name = class_names_from_model[cls_id]
                            text_to_draw = f"{pred_class_name}: {conf:.2f}"

                            x1, y1, x2, y2 = coords
                            color = (0, 255, 0) # Mặc định xanh lá
                            if pred_class_name == 'DRONE': color = (0, 255, 0)
                            elif pred_class_name == 'HELICOPTER': color = (0, 0, 255)
                            elif pred_class_name == 'AIRPLANE': color = (255, 0, 0)
                            elif pred_class_name == 'BIRD': color = (0, 255, 255)

                            cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(drawn_frame, text_to_draw, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Lưu frame đã annotate (nếu có phát hiện đáng kể hoặc luôn lưu)
                # Ở đây, chúng ta sẽ lưu frame nếu có ít nhất một phát hiện được vẽ
                # Hoặc bạn có thể quyết định lưu tất cả các frame đã xử lý (thay đổi điều kiện if)
                if found_detection_to_draw: # Chỉ lưu nếu có gì đó được vẽ (conf > conf_thresh_viz)
                    annotated_filename = f"{os.path.splitext(video_filename)[0]}_frame_{frame_count:06d}_annotated.jpg"
                    save_path = os.path.join(output_class_annotated_dir, annotated_filename)
                    cv2.imwrite(save_path, drawn_frame)
                    processed_frame_count += 1
                # else: # Nếu bạn muốn lưu cả frame không có detection cao
                #     annotated_filename = f"{os.path.splitext(video_filename)[0]}_frame_{frame_count:06d}_no_high_conf.jpg"
                #     save_path = os.path.join(output_class_annotated_dir, annotated_filename)
                #     cv2.imwrite(save_path, frame) # Lưu frame gốc

            cap.release()
            # print(f"  Đã xử lý video {video_filename}, lưu {processed_frame_count} frame có chú thích.")

    print("\n===== ANNOTATING OOD VIDEOS FINISHED =====")

if __name__ == '__main__':
    # Đảm bảo các thư mục output được tạo nếu hàm ensure_output_directories không được gọi tự động khi import config
    if hasattr(cfg_annotate, 'ensure_output_directories'):
        cfg_annotate.ensure_output_directories()
    else:
        os.makedirs(cfg_annotate.EVALUATION_OOD_ANNOTATED_IMAGES_DIR, exist_ok=True)
        for class_n in cfg_annotate.MASTER_CLASS_LIST_FUSION: # Giả sử bạn muốn tạo thư mục con theo lớp
             os.makedirs(os.path.join(cfg_annotate.EVALUATION_OOD_ANNOTATED_IMAGES_DIR, class_n), exist_ok=True)


    annotate_videos_from_ood_set(
        ood_video_base_dir=cfg_annotate.EVALUATION_OOD_RAW_VIDEO_BASE_DIR,
        output_annotated_base_dir=cfg_annotate.EVALUATION_OOD_ANNOTATED_IMAGES_DIR,
        vcam_model_path=cfg_annotate.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH,
        img_size=cfg_annotate.VCAM_YOUTUBE_FINETUNE_IMG_SIZE, # Sử dụng imgsz khi fine-tune
        conf_thresh_viz=0.3, # Ngưỡng confidence để vẽ box, có thể điều chỉnh
        frame_skip=30 # Ví dụ: xử lý 1 frame mỗi giây nếu video 30fps
    )