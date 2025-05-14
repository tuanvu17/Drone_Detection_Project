# Drone_Detection_Project/src/inference/predict_ir_video.py
import os
import glob
import cv2 # Thư viện OpenCV để xử lý video và ảnh
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image # Dùng để mở ảnh nếu transform cần

# THAY THẾ BẰNG ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH IRCAM CỦA BẠN
# from src.model_architectures.ircam_pytorch_model import YourIRCamModelDefinition
# Ví dụ một định nghĩa giả định:
class YourIRCamModelDefinition(torch.nn.Module):
    def __init__(self, num_classes):
        super(YourIRCamModelDefinition, self).__init__()
        # Đây chỉ là ví dụ, bạn cần thay thế bằng kiến trúc thực tế của bạn
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1) # Giả sử ảnh IR 1 kênh
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(32 * (224//4) * (224//4), 128) # Kích thước phụ thuộc vào input và các lớp conv/pool
        self.fc2 = torch.nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# KẾT THÚC THAY THẾ

# --- Cấu hình (Có thể lấy từ file config nếu bạn đã tích hợp) ---
# Nên lấy từ file config để nhất quán
try:
    from src.config_loader.loader import get_config
    cfg = get_config()
    # Giả sử bạn có các cấu hình sau trong project_config.py
    IR_VIDEO_TO_PREDICT_DIR = os.path.join(cfg.PROJECT_ROOT, 'data', 'raw', 'IR_Videos_Original', 'VIDEO2PREDICT')
    IR_MODEL_PATH = os.path.join(cfg.PROJECT_ROOT, 'models', 'ircam_classifier', 'ir_cam_best.pt') # THAY TÊN FILE NẾU CẦN
    IR_NUM_CLASSES = 3 # THAY ĐỔI SỐ LỚP PHÙ HỢP VỚI MÔ HÌNH IR CỦA BẠN
    IR_CLASS_NAMES = cfg.CLASSES_ALL # Hoặc một list tên lớp riêng cho IR
    IR_IMAGE_SIZE = (224, 224) # Kích thước ảnh đầu vào của mô hình IR
    IR_FRAMES_PER_SEGMENT = 5 # Số frame lấy từ mỗi giây để dự đoán (trung bình)
except ImportError:
    print("Cảnh báo: Không thể tải config, sử dụng giá trị mặc định.")
    # Giá trị mặc định nếu không có config
    PROJECT_ROOT_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    IR_VIDEO_TO_PREDICT_DIR = os.path.join(PROJECT_ROOT_DEFAULT, 'data', 'raw', 'IR_Videos_Original', 'VIDEO2PREDICT')
    IR_MODEL_PATH = os.path.join(PROJECT_ROOT_DEFAULT, 'models', 'ircam_classifier', 'ir_cam_best.pt')
    IR_NUM_CLASSES = 3
    IR_CLASS_NAMES = ['BACKGROUND', 'DRONE', 'HELICOPTER'] # Ví dụ
    IR_IMAGE_SIZE = (224, 224)
    IR_FRAMES_PER_SEGMENT = 5


def preprocess_ir_frame(frame, image_size):
    """Tiền xử lý một frame ảnh IR."""
    # Chuyển sang ảnh xám nếu là ảnh màu (ảnh IR thường là ảnh xám)
    if frame.shape[2] == 3: # Nếu có 3 kênh màu
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    # Resize ảnh
    resized_frame = cv2.resize(frame_gray, image_size, interpolation=cv2.INTER_AREA)

    # Chuẩn hóa (ví dụ: về [0, 1] rồi chuẩn hóa theo mean/std của ImageNet nếu mô hình dựa trên đó)
    # Hoặc chuẩn hóa theo cách bạn đã làm khi huấn luyện mô hình IR
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Ví dụ chuẩn hóa, bạn cần điều chỉnh cho phù hợp với mô hình IR của mình
        transforms.Normalize(mean=[0.5], std=[0.5]) # Giả sử ảnh 1 kênh và chuẩn hóa đơn giản
    ])
    return transform(resized_frame)


def predict_ir_video(video_path, model, device, class_names, image_size, frames_per_segment):
    """Dự đoán trên từng frame của video IR và tổng hợp kết quả."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video: {video_path}")
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Cảnh báo: Không thể lấy FPS cho video {video_path}. Sử dụng mặc định 25 FPS.")
        fps = 25 # Giá trị mặc định nếu không lấy được

    frame_count = 0
    segment_predictions = [] # Lưu trữ dự đoán của các segment (mỗi segment là nhiều frame)

    print(f"Đang xử lý video: {os.path.basename(video_path)} (FPS: {fps:.2f})")

    current_segment_frames_processed = 0
    current_segment_outputs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Hết video

        frame_count += 1
        current_segment_frames_processed += 1

        # Tiền xử lý frame
        try:
            input_tensor = preprocess_ir_frame(frame, image_size).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  Lỗi tiền xử lý frame {frame_count}: {e}")
            continue

        # Dự đoán
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            current_segment_outputs.append(probabilities.cpu().numpy().flatten())

        # Khi đủ số frame cho một segment hoặc hết video
        if current_segment_frames_processed >= frames_per_segment or not ret: # Thêm điều kiện not ret để xử lý segment cuối
            if current_segment_outputs:
                avg_segment_probs = np.mean(np.array(current_segment_outputs), axis=0)
                segment_predictions.append(avg_segment_probs)
                # predicted_class_idx = np.argmax(avg_segment_probs)
                # predicted_class_name = class_names[predicted_class_idx]
                # confidence = avg_segment_probs[predicted_class_idx]
                # print(f"  Segment (frames ~{frame_count-frames_per_segment+1}-{frame_count}): Dự đoán {predicted_class_name}, Độ tự tin: {confidence:.2f}")

            current_segment_frames_processed = 0
            current_segment_outputs = []
            if not ret: # Nếu là do hết video
                break


    cap.release()

    if not segment_predictions:
        print(f"  Không có dự đoán nào được thực hiện cho video {video_path}.")
        return None, None, None

    # Tổng hợp kết quả cho toàn bộ video (ví dụ: lấy trung bình xác suất của các segment)
    final_video_probs = np.mean(np.array(segment_predictions), axis=0)
    final_predicted_idx = np.argmax(final_video_probs)
    final_predicted_class = class_names[final_predicted_idx]
    final_confidence = float(final_video_probs[final_predicted_idx])

    return final_predicted_class, final_confidence, final_video_probs.tolist()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # Tải mô hình IR
    try:
        ir_model = YourIRCamModelDefinition(num_classes=IR_NUM_CLASSES) # THAY THẾ BẰNG KIẾN TRÚC CỦA BẠN
        ir_model.load_state_dict(torch.load(IR_MODEL_PATH, map_location=device))
        ir_model.to(device)
        ir_model.eval() # Chuyển sang chế độ đánh giá
        print(f"Tải mô hình IR thành công từ: {IR_MODEL_PATH}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình IR: {e}")
        exit()

    video_files = glob.glob(os.path.join(IR_VIDEO_TO_PREDICT_DIR, "*")) # Lấy tất cả file
    video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]


    if not video_files:
        print(f"Không tìm thấy file video nào trong: {IR_VIDEO_TO_PREDICT_DIR}")
    else:
        print(f"\n--- Bắt đầu dự đoán trên các video IR trong: {IR_VIDEO_TO_PREDICT_DIR} ---")
        for video_path in video_files:
            pred_class, confidence, probs = predict_ir_video(
                video_path, ir_model, device, IR_CLASS_NAMES, IR_IMAGE_SIZE, IR_FRAMES_PER_SEGMENT
            )
            if pred_class:
                print(f"\nVideo: {os.path.basename(video_path)}")
                print(f"  Dự đoán cuối cùng cho video: {pred_class}")
                print(f"  Độ tự tin: {confidence:.4f}")
                print(f"  Xác suất các lớp ({IR_CLASS_NAMES}):")
                for i, class_name in enumerate(IR_CLASS_NAMES):
                    print(f"    - {class_name}: {probs[i]:.4f}")
            else:
                print(f"\nKhông thể xử lý video: {os.path.basename(video_path)}")