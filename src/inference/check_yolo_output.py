# Drone_Detection_Project/src/inference/check_yolo_output.py
import os
import cv2
from ultralytics import YOLO
import torch # Cần để set device nếu muốn

# Import config
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    PROJECT_ROOT_CHECK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT_CHECK not in sys.path:
        sys.path.append(PROJECT_ROOT_CHECK)
    from config import project_config as cfg_default # Tạm dùng tên khác để tránh nhầm lẫn nếu file này được gọi từ đâu đó
else:
    cfg_default = get_config()


def inspect_yolo_model_output(model_path, sample_image_path, imgsz=640, conf_thresh=0.25):
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại: {model_path}")
        return
    if not os.path.exists(sample_image_path):
        print(f"Lỗi: Không tìm thấy file ảnh mẫu tại: {sample_image_path}")
        return

    print(f"Tải mô hình YOLO từ: {model_path}")
    try:
        model = YOLO(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # Đảm bảo model ở đúng device
        print(f"  Mô hình được tải lên thiết bị: {model.device}")
        print(f"  Tên các lớp của mô hình: {model.names}") # QUAN TRỌNG
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    print(f"\nĐọc ảnh mẫu từ: {sample_image_path}")
    frame = cv2.imread(sample_image_path)
    if frame is None:
        print(f"Lỗi: Không thể đọc ảnh mẫu.")
        return

    print(f"Thực hiện dự đoán trên ảnh mẫu (imgsz={imgsz}, conf={conf_thresh})...")
    # Chạy predict
    results = model.predict(source=frame, imgsz=imgsz, conf=conf_thresh, verbose=True) # verbose=True để xem log

    # Phân tích output
    if results and isinstance(results, list) and len(results) > 0:
        result = results[0] # Lấy kết quả cho ảnh đầu tiên (và duy nhất)
        print("\n--- Thông tin Đối tượng Results ---")
        print(f"Số lượng box phát hiện được: {len(result.boxes)}")

        if result.boxes:
            print("\n--- Chi tiết từng Bounding Box ---")
            for i in range(len(result.boxes)):
                box_data = result.boxes[i]
                xyxy = box_data.xyxy[0].cpu().numpy() # Tọa độ [x1, y1, x2, y2]
                confidence = float(box_data.conf[0])
                class_id = int(box_data.cls[0])
                class_name = model.names[class_id] # Lấy tên lớp từ thuộc tính names của model

                print(f"  Box {i+1}:")
                print(f"    Tọa độ (xyxy): {xyxy}")
                print(f"    Lớp ID: {class_id}, Tên lớp: {class_name}")
                print(f"    Độ tự tin: {confidence:.4f}")
                # Kiểm tra xem có output xác suất cho tất cả các lớp không
                # Một số phiên bản/cách gọi YOLO có thể không trả về probs trực tiếp trong box_data
                # mà nằm trong result.probs hoặc cần lấy từ model(image).logits
                if hasattr(box_data, 'probs') and box_data.probs is not None:
                    print(f"    Xác suất lớp (nếu có): {box_data.probs.data[0].cpu().numpy()}")
                elif hasattr(result, 'probs') and result.probs is not None: # Thường cho classification model
                     print(f"    Xác suất lớp (từ result.probs): {result.probs.data[0].cpu().numpy()}")
                else:
                    print(f"    Không tìm thấy vector xác suất lớp chi tiết trong box_data hoặc result.probs.")


        # In toàn bộ đối tượng result để xem thêm
        # print("\n--- Toàn bộ đối tượng Result[0] ---")
        # print(result)

        # Vẽ và hiển thị (tùy chọn)
        annotated_frame = result.plot()
        cv2.imshow("YOLO Detection Result", annotated_frame)
        print("\nNhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Không có kết quả dự đoán nào được trả về.")

if __name__ == "__main__":
    # Sử dụng đường dẫn từ config cho mô hình VCam đã fine-tune trên YouTube
    model_to_check_path = cfg_default.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH # VCAM_YOLO_BEST_MODEL_SAVE_PATH
    img_size_to_check = cfg_default.VCAM_YOUTUBE_FINETUNE_IMG_SIZE

    # BẠN CẦN CUNG CẤP ĐƯỜNG DẪN ĐẾN MỘT ẢNH MẪU ĐỂ TEST
    # Ví dụ: lấy một frame từ video test của bạn
    sample_image_for_yolo_check = os.path.join(cfg_default.PROJECT_ROOT, "data", "raw", "YouTube_Videos_Raw", "VIDEO2PREDICT", "V_HELICOPTER_002_frame_0017.jpg")
    # Hãy đảm bảo file ảnh này tồn tại, hoặc thay bằng đường dẫn của bạn
    if not os.path.exists(sample_image_for_yolo_check):
        print(f"LƯU Ý: File ảnh mẫu '{sample_image_for_yolo_check}' không tồn tại.")
        print("Vui lòng tạo một ảnh mẫu (ví dụ: một frame từ video test) và cập nhật đường dẫn trên.")
        print("Hoặc bạn có thể comment out phần gọi hàm inspect_yolo_model_output.")
    else:
        inspect_yolo_model_output(model_to_check_path, sample_image_for_yolo_check, imgsz=img_size_to_check)