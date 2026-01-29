# Drone_Detection_Project/src/main_experiment_compare_features.py
"""
Script chính để chạy thực nghiệm so sánh các đặc trưng audio
Chạy: python -m src.main_experiment_compare_features
"""

import sys
import os

# Thêm đường dẫn project vào sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.compare_audio_features import main

if __name__ == "__main__":
    main()

