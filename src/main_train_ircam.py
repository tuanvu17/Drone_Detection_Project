# Drone_Detection_Project/src/main_train_ircam.py
import sys
import os
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.append(PROJECT_ROOT_DIR)

from src.training.train_ircam_yolo import run_ircam_yolo_training

if __name__ == '__main__':
    run_ircam_yolo_training()