# Drone_Detection_Project/src/config_loader/loader.py
# Drone_Detection_Project/src/config_loader/loader.py
import sys
import os

def get_config():
    try:
        # Tìm đường dẫn đến thư mục config từ vị trí hiện tại của loader.py
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/config_loader
        src_dir = os.path.dirname(current_dir) # src
        project_root_from_loader = os.path.dirname(src_dir) # Drone_Detection_Project
        config_file_dir = os.path.join(project_root_from_loader, 'config')

        if config_file_dir not in sys.path:
            sys.path.insert(0, config_file_dir) # Thêm thư mục 'config' vào đầu sys.path

        import project_config as cfg # Bây giờ Python sẽ tìm thấy project_config.py
        if hasattr(cfg, 'ensure_output_directories'):
            cfg.ensure_output_directories()
        return cfg
    except ImportError as e:
        print(f"Lỗi: Không thể import project_config.py. Đảm bảo nó nằm trong thư mục 'config' ở gốc dự án.")
        print(f"Chi tiết lỗi: {e}")
        print(f"sys.path hiện tại: {sys.path}")
        # Thử in ra đường dẫn mong đợi để debug
        expected_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))), 'config')
        print(f"Đường dẫn mong đợi của thư mục config: {expected_config_path}")

        raise # Ném lại lỗi để dừng chương trình nếu không tải được config
    except Exception as e:
        print(f"Lỗi không xác định khi tải cấu hình: {e}")
        raise

# Tạo các file __init__.py trống trong các thư mục con của src nếu chưa có:
# src/__init__.py
# src/config_loader/__init__.py
# src/data_processing/__init__.py
# src/model_architectures/__init__.py
# src/training/__init__.py
# src/evaluation/__init__.py