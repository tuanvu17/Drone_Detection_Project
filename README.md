
Drone_Detection_Project/
📁 Drone_Detection_Project/
├── 📄 .gitignore
├── ℹ️ README.md
├── ⚙️ config/
│   ├── 🐍 __init__.py
│   └── 📜 project_config.py
│
├── 📦 data/
│   ├── ⏳ interim/                     # Dữ liệu trung gian
│   │   └── 🔊 YouTube_Audio_Extracted/
│   │       └── 🎵 drone_flight_01_audio.wav
│   ├── ✨ processed/                  # Dữ liệu đã xử lý
│   │   ├── 🎶 Audio_Segments_MFCC/
│   │   │   ├── 🧪 test/
│   │   │   │   ├── 📁 BACKGROUND/
│   │   │   │   ├── 🚁 DRONE/
│   │   │   │   └── 🚁 HELICOPTER/
│   │   │   ├── 🏋️ train/
│   │   │   │   ├── 📁 BACKGROUND/
│   │   │   │   ├── 🚁 DRONE/
│   │   │   │   └── 🚁 HELICOPTER/
│   │   │   └── ☑️ validation/
│   │   │       ├── 📁 BACKGROUND/
│   │   │       ├── 🚁 DRONE/
│   │   │       └── 🚁 HELICOPTER/
│   │   ├── 🖼️ FineTune_Paired_Data/      # Dữ liệu ghép cặp ảnh-âm thanh
│   │   │   └── 🚁 DRONE/
│   │   │       ├── 🏞️ video1_seg001_frame.jpg
│   │   │       └── 🎼 video1_seg001_audio_mfcc.npy
│   │   ├── 🖼️ VCam_Processed_Frames/
│   │   │   ├── 🧪 test/
│   │   │   ├── 🏋️ train/
│   │   │   └── ☑️ validation/
│   │   └── 📋 test_files_list_audio.txt
│   │
│   └── 📀 raw/                       # Dữ liệu thô
│       ├── 🔊 Audio_Original/
│       │   ├── 📁 BACKGROUND/
│       │   │   └── 🎵 background_001.wav
│       │   ├── 🚁 DRONE/
│       │   │   └── 🎵 drone_001.wav
│       │   └── 🚁 HELICOPTER/
│       │       └── 🎵 helicopter_001.wav
│       ├── 🔥 IRCam_Images_Original/    # Ảnh nhiệt thô
│       │   └── 🚁 DRONE/
│       ├── 📷 VCam_Images_Original/     # Ảnh thường thô
│       │   └── 🚁 DRONE/
│       └── 🎞️ YouTube_Videos_Raw/
│           └── 🎬 drone_flight_01.mp4
│
├── 🧠 models/                      # Mô hình đã huấn luyện
│   ├── 🎤 audio_classifier/
│   │   ├── 🏷️ audio_test_class_names.pkl
│   │   ├── 🔮 audio_test_predictions.pkl
│   │   ├── ✅ audio_test_true_labels.pkl
│   │   ├── 💾 best_audio_model.keras
│   │   ├── 🏷️ label_encoder_audio.pkl
│   │   ├── 📏 max_len_audio.pkl
│   │   └── ⚖️ scaler_audio.pkl
│   ├── 🧩 fine_tuned_fusion_model/
│   ├── 🔥 ircam_classifier/
│   └── 📷 vcam_classifier/
│
├── 📓 notebooks/                   # Jupyter notebooks
│   ├──  exploratory_analysis 01_data_exploration.ipynb
│   ├── 🏋️ 02_audio_model_training.ipynb
│   └── 🔧 03_vcam_audio_finetuning.ipynb
│
├── 📊 reports/                     # Báo cáo và kết quả
│   ├── 🖼️ figures/                 # Hình ảnh, biểu đồ
│   │   ├── 📉 audio_confusion_matrix_test.png
│   │   ├── 📉 audio_confusion_matrix_val.png
│   │   └── 📈 audio_training_history.png
│   └── 📄 metrics/                 # Số liệu chi tiết
│       └── 📝 audio_detailed_test_results.txt
│
├── 📜 requirements.txt
│
└── 🐍 src/                         # Mã nguồn
    ├── 🐍 __init__.py
    ├── ⚙️ config_loader/
    │   ├── 🐍 __init__.py
    │   └── 📜 loader.py
    ├── 🛠️ data_processing/
    │   ├── 🐍 __init__.py
    │   ├── 🎶 audio_utils.py
    │   ├── 🖼️ image_utils.py
    │   └── 🎞️ video_utils.py
    ├── 🧪 evaluation/
    │   ├── 🐍 __init__.py
    │   ├── ⚖️ evaluate_audio.py
    │   └── 📈 plotting_utils.py
    ├── 🏗️ model_architectures/
    │   ├── 🐍 __init__.py
    │   ├── 🎤 audio_model.py
    │   ├── 🧩 fusion_model.py
    │   └── 📷 vcam_model.py
    ├── 🏋️ training/
    │   ├── 🐍 __init__.py
    │   ├── 🔧 fine_tune_fusion.py
    │   └── 🚂 train_audio_pipeline.py
    ├── ▶️ main_finetune_fusion.py
    ├── ▶️ main_train_audio.py
    └── 🚀 predict.py
    └── inference/              # THƯ MỤC MỚI
        ├── __init__.py         # File trống
        └── predict_audio.py    # FILE MỚI



Hướng dẫn cách chạy:
Đảm bảo các file __init__.py:
Tạo file __init__.py (có thể trống) trong các thư mục sau nếu chưa có:
Drone_Detection_Project/src/__init__.py
Drone_Detection_Project/src/config_loader/__init__.py
Drone_Detection_Project/src/data_processing/__init__.py
Drone_Detection_Project/src/model_architectures/__init__.py
Drone_Detection_Project/src/training/__init__.py
Drone_Detection_Project/src/evaluation/__init__.py
Drone_Detection_Project/config/__init__.py (QUAN TRỌNG để from config import project_config hoạt động)
Mở Terminal hoặc Command Prompt.
Di chuyển đến thư mục gốc của dự án:
cd /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project

# python -m src.main_train_audio


Predict file audio test:
# python -m src.inference.predict_audio


train ir cam 
# python -m src.main_train_ircam


train v cam 
# python -m src.main_train_vcam


cd /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project
# python -m src.inference.predict_ircam_yolo_video

# python -m src.inference.predict_vcam_yolo_video

Predict file v_cam test:
# python -m src.inference.predict_vcam_yolo_video


Prepare data for Fusion:
# python -m src.data_processing.prepare_fusion_finetune_data

Kiểm tra số lượng cặp ghép: 
# python -m ultils.analyze_fusion_paired_data


Train Fusion 
# python -m src.main_finetune_fusion

Predict Fusion vcam audio
# python -m src.inference.predict_fusion_video

Chạy Script Chuẩn bị Dữ liệu Audio Segment:
Đứng từ thư mục Drone_Detection_Project
# python -m src.data_processing.prepare_audio_youtube_data

Chạy Script Fine-tune Mô hình Audio:
Đứng từ thư mục Drone_Detection_Project
# python -m src.main_finetune_audio_youtube

lây thông tin fine-tune Audio:
# python -m ultils.analyze_audio_dataset

Thực hiện chạy Test OOD
Chuẩn bị dữ liệu cho OOD: 
# python -m src.data_processing.prepare_ood_evaluation_data


# python -m src.main_evaluate_comparison


Thực hiện chạy Test In-Domain
# python -m src.main_evaluate_comparison_in_domain


bổ sung background cho vcam fine tune yolo
# python -m src.ultils.add_background_frames_to_yolo
# python -m ultils.analyze_yolo_dataset

train mô hình vcam fine tune
# python -m src.main_finetune_vcam_youtube




thực hiện test trên miền Out - of - domain
Chuẩn bị Dataset cho Testing
# python -m src.data_processing.prepare_evaluation_test_data
# python -m src.data_processing.prepare_ood_evaluation_data

Thực hiện chạy test 
# python -m src.main_evaluate_comparison


Kiểm tra tên lớp để trích xuất Output cho Audio:
# python -m ultils.check_audio_model_summary




**Tóm tắt nhanh — các bước chạy chính trong Drone_Detection_Project**

- **Chuẩn bị môi trường**: cài dependencies và đứng ở thư mục gốc dự án.
  - Cài đặt:
    ```bash
    cd /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project
    pip install -r requirements.txt
    ```
  - Đảm bảo các `__init__.py` (README ghi rõ danh sách thư mục cần có file này).

**1) Chuẩn bị dữ liệu**
- **Audio segments**:
  - Tạo/chuẩn hóa segment và MFCC:
    ```bash
    python -m src.data_processing.prepare_audio_youtube_data
    ```
- **VCam frames / YOLO dataset**:
  - Các script tiền xử lý nằm trong `src/data_processing` (xem `image_utils.py`, notebooks).
- **Dữ liệu cho Fusion (paired image+audio)**:
  - Chuẩn bị cặp để fine-tune fusion:
    ```bash
    python -m src.data_processing.prepare_fusion_finetune_data
    ```
- Kiểm tra số cặp ghép:
  ```bash
  python -m ultils.analyze_fusion_paired_data
  ```

**2) Huấn luyện mô hình Audio**
- Chạy pipeline huấn luyện audio:
  ```bash
  python -m src.main_train_audio
  ```
- Fine-tune audio (YouTube / dataset khác):
  ```bash
  python -m src.main_finetune_audio_youtube
  ```
- Kiểm tra thông tin dataset/fine-tune:
  ```bash
  python -m ultils.analyze_audio_dataset
  ```

**3) Huấn luyện VCam / IRCam**
- Train VCam classifier:
  ```bash
  python -m src.main_train_vcam
  ```
- Train IR cam (nếu cần):
  ```bash
  python -m src.main_train_ircam
  ```
- Fine-tune VCam (YOLO/background augmentation flow):
  - Thêm background frames:
    ```bash
    python -m src.ultils.add_background_frames_to_yolo
    python -m ultils.analyze_yolo_dataset
    ```
  - Fine-tune YOLO/VCam:
    ```bash
    python -m src.main_finetune_vcam_youtube
    ```

**4) Huấn luyện / Fine-tune Fusion (image + audio)**
- Chuẩn bị cặp (như mục ở trên), rồi huấn luyện fusion:
  ```bash
  python -m src.main_finetune_fusion
  ```
- Mô hình fusion lưu ở `models/fine_tuned_fusion_model/` theo README.

**5) Inference / Predict**
- Dự đoán audio (file test):
  ```bash
  python -m src.inference.predict_audio
  ```
- Dự đoán VCam video (YOLO):
  ```bash
  python -m src.inference.predict_vcam_yolo_video
  ```
- Dự đoán IRcam video:
  ```bash
  python -m src.inference.predict_ircam_yolo_video
  ```
- Dự đoán Fusion (video):
  ```bash
  python -m src.inference.predict_fusion_video
  ```

**6) Chuẩn bị và chạy đánh giá (Evaluation)**
- Chuẩn bị dữ liệu OOD / test:
  ```bash
  python -m src.data_processing.prepare_ood_evaluation_data
  # python -m src.data_processing.prepare_evaluation_test_data
  ```
- Chạy so sánh / đánh giá:
  - Out-of-domain comparison:
    ```bash
    python -m src.main_evaluate_comparison
    ```
  - In-domain comparison:
    ```bash
    python -m src.main_evaluate_comparison_in_domain
    ```
- Các script đánh giá audio cụ thể:
  ```bash
  python -m src.evaluation.evaluate_audio
  ```
- Kết quả/metrics xuất ra thư mục `reports/metrics/` theo README.

**7) Kiểm tra/Phân tích bổ sung**
- Kiểm tra lớp đầu ra audio:
  ```bash
  python -m ultils.check_audio_model_summary
  ```
- Các notebook phân tích / visual có trong `notebooks/` (ví dụ training history, conf_matrix).

**Gợi ý đánh giá & metrics cần thu**
- Object detection (VCam/IR): mAP@0.5, Precision, Recall, IoU, FPS/latency.
- Audio classification: Accuracy, Precision/Recall, F1, Confusion Matrix.
- Fusion: mAP/accuracy trên dataset ghép, robustness khi một modal nhiễu (ablation).
- Lưu logs/weights: kiểm tra `models/` và `reports/metrics/` để so sánh phiên bản.

**Lưu ý thực tiễn**
- Luôn chạy từ thư mục gốc dự án (README khuyến cáo).
- Kiểm tra `config/project_config.py` để biết đường dẫn file, hyperparams, checkpoint tên file.
- Nếu cần chạy trên GPU, xác nhận CUDA/torch setup trước khi train.
- Nên tạo script shell hoặc Makefile để tự động hóa chuỗi: data -> train audio -> train vcam -> prepare fusion -> train fusion -> eval.

Muốn tôi:
- Tạo một `run_all.sh` (hoặc `Makefile`) tự động hóa chuỗi chạy này không?
- Hoặc tôi cài đặt một module Cross-Attention fusion vào fusion_model.py và tạo ví dụ forward/test?