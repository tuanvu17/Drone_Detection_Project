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




# Drone_Detection_Project
