# Drone_Detection_Project/src/main_generate_audio_results.py
"""
Script chính để tạo báo cáo kết quả và phân tích cho các mô hình Audio
Chạy: python -m src.main_generate_audio_results
"""

import sys
import os

# Thêm đường dẫn project vào sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.generate_audio_results_report import main

if __name__ == "__main__":
    main()

