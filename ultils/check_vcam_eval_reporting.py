# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/check_vcam_eval_reporting.py
import os
import yaml
import glob
import random
from ultralytics import YOLO
import torch
import numpy as np # Thêm numpy

# Import config
try:
    from src.config_loader.loader import get_config
    cfg_check = get_config()
except ImportError:
    import sys
    current_dir_check = os.path.dirname(os.path.abspath(__file__))
    src_dir_check = os.path.dirname(current_dir_check)
    project_root_check = os.path.dirname(src_dir_check)
    if project_root_check not in sys.path: sys.path.insert(0, project_root_check)
    if src_dir_check not in sys.path: sys.path.insert(0, src_dir_check)
    from config_loader.loader import get_config

cfg_check = get_config() # Đảm bảo cfg_check được gán

def check_vcam_evaluation_reporting_logic(num_sample_images_to_test=5):
    print("===== STARTING VCAM EVALUATION REPORTING CHECK =====")

    # 1. Kiểm tra các file cấu hình và mô hình gốc
    if not os.path.exists(cfg_check.VCAM_YOUTUBE_DATA_YAML_PATH):
        print(f"LỖI: Không tìm thấy file data.yaml: {cfg_check.VCAM_YOUTUBE_DATA_YAML_PATH}")
        return
    if not os.path.exists(cfg_check.VCAM_YOLO_BEST_MODEL_SAVE_PATH): # Model VCam gốc
        print(f"LỖI: Không tìm thấy mô hình VCam gốc: {cfg_check.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
        return

    # 2. Đọc class names từ file YAML
    class_names_from_yaml = []
    path_to_yaml_dir = ""
    test_images_path_from_yaml = ""
    try:
        with open(cfg_check.VCAM_YOUTUBE_DATA_YAML_PATH, 'r') as f_yaml:
            data_yaml_content = yaml.safe_load(f_yaml)
            path_to_yaml_dir = os.path.dirname(cfg_check.VCAM_YOUTUBE_DATA_YAML_PATH) # Thư mục chứa file yaml

            if 'names' in data_yaml_content and isinstance(data_yaml_content['names'], list):
                class_names_from_yaml = data_yaml_content['names']
                print(f"Các lớp được định nghĩa trong YAML: {class_names_from_yaml} (Tổng: {len(class_names_from_yaml)} lớp)")
            else:
                print(f"CẢNH BÁO: Không tìm thấy 'names' hoặc 'names' không phải là list trong YAML.")
                return

            # Xác định đường dẫn đến thư mục test/images
            test_path_relative = data_yaml_content.get('test', None)
            if test_path_relative:
                 # Nếu đường dẫn trong yaml là tương đối với thư mục chứa file yaml
                if not os.path.isabs(test_path_relative):
                    test_images_path_from_yaml = os.path.join(path_to_yaml_dir, test_path_relative)
                else: # Nếu là đường dẫn tuyệt đối
                    test_images_path_from_yaml = test_path_relative
                
                # YOLO thường mong đợi đường dẫn đến thư mục images, không phải file txt
                # Nếu test_path_relative là file .txt, ta cần lấy thư mục images tương ứng
                if os.path.isfile(test_images_path_from_yaml) and test_images_path_from_yaml.endswith('.txt'):
                    # Giả sử cấu trúc: data_root/test.txt, data_root/images/
                    # Hoặc data_root/dataset_name/test.txt, data_root/dataset_name/images/
                    # Cách an toàn hơn là file yaml nên trỏ trực tiếp đến thư mục dataset gốc
                    # và 'test' key trong yaml là 'test/images'
                    print(f"Cảnh báo: Đường dẫn 'test' trong YAML ('{test_path_relative}') trỏ đến file text. ")
                    print(f"  Script này giả định 'test' key trong YAML nên là 'test/images' (tương đối với path trong YAML)")
                    print(f"  Hoặc path trong YAML là thư mục gốc của dataset (VD: VCam_YouTube_FineTune_Data) và test là 'test/images'")
                    # Cố gắng tìm thư mục images dựa trên cấu trúc phổ biến
                    potential_images_dir = os.path.join(os.path.dirname(cfg_check.VCAM_YOUTUBE_DATA_YAML_PATH), "test", "images")
                    if os.path.isdir(potential_images_dir):
                        test_images_path_from_yaml = potential_images_dir
                        print(f"  Đã tự động điều chỉnh đường dẫn test images thành: {test_images_path_from_yaml}")
                    else:
                        print(f"  LỖI: Không thể tự động xác định thư mục test/images từ YAML.")
                        return

                elif not test_images_path_from_yaml.endswith("images"):
                    # Nếu là thư mục, nhưng không phải images, ví dụ chỉ là "test"
                    test_images_path_from_yaml = os.path.join(test_images_path_from_yaml, "images")


                if not os.path.isdir(test_images_path_from_yaml):
                    print(f"LỖI: Thư mục test images ('{test_images_path_from_yaml}') không tồn tại.")
                    return
            else:
                print("LỖI: Không có key 'test' trong file YAML để xác định dữ liệu test.")
                return

    except Exception as e_yaml:
        print(f"Lỗi khi đọc hoặc xử lý file YAML: {e_yaml}")
        return

    # 3. Tải mô hình VCam (có thể là gốc hoặc fine-tuned, ở đây dùng gốc để test logic)
    print(f"\nĐang tải mô hình VCam (ví dụ: mô hình gốc) từ: {cfg_check.VCAM_YOLO_BEST_MODEL_SAVE_PATH}")
    try:
        # Sử dụng mô hình gốc để kiểm tra, vì mô hình fine-tuned có thể chưa có
        model_to_check = YOLO(cfg_check.VCAM_YOLO_BEST_MODEL_SAVE_PATH)
        device_to_use = '0' if torch.cuda.is_available() else 'cpu'
        # model_to_check.to(device_to_use) # Không cần thiết, sẽ truyền vào val
        print(f"Đã tải mô hình. Device sẽ dùng: {device_to_use}")
    except Exception as e_model:
        print(f"Lỗi khi tải mô hình VCam: {e_model}")
        return

    # Lấy tên lớp từ model đã tải (để so sánh với YAML)
    model_class_names = []
    if hasattr(model_to_check, 'names'):
        if isinstance(model_to_check.names, dict):
            model_class_names = [model_to_check.names[i] for i in sorted(model_to_check.names.keys())]
        elif isinstance(model_to_check.names, list):
            model_class_names = model_to_check.names
    print(f"Các lớp trong model.names: {model_class_names} (Tổng: {len(model_class_names)} lớp)")
    if not model_class_names:
        print("CẢNH BÁO: Không thể lấy danh sách lớp từ model.names.")
    elif set(model_class_names) != set(class_names_from_yaml):
        print("CẢNH BÁO: Danh sách lớp trong model.names không hoàn toàn khớp với 'names' trong YAML!")
        print(f"  YAML names: {class_names_from_yaml}")
        print(f"  Model names: {model_class_names}")
        print("  Điều này có thể dẫn đến kết quả mAP không chính xác hoặc lỗi khi huấn luyện/đánh giá.")
        print("  Hãy đảm bảo file YAML được dùng để huấn luyện model này có 'names' khớp với model.names.")


    # 4. Mô phỏng chạy model.val() trên một vài ảnh mẫu từ tập test
    print(f"\n--- Mô phỏng chạy model.val() trên một vài ảnh từ: {test_images_path_from_yaml} ---")
    all_test_images = glob.glob(os.path.join(test_images_path_from_yaml, '*.jpg')) + \
                      glob.glob(os.path.join(test_images_path_from_yaml, '*.png')) + \
                      glob.glob(os.path.join(test_images_path_from_yaml, '*.jpeg'))

    if not all_test_images:
        print(f"LỖI: Không tìm thấy ảnh nào trong thư mục test: {test_images_path_from_yaml}")
        return

    # Chọn ngẫu nhiên một vài ảnh để test (hoặc toàn bộ nếu tập test nhỏ)
    # Để mô phỏng nhanh, ta sẽ không chạy .val() thực sự vì nó cần cả thư mục labels
    # Thay vào đó, ta sẽ giả lập cấu trúc output của test_metrics.box.maps
    print("Giả lập kết quả `test_metrics.box.maps` (mAP50-95 per class):")
    # maps_per_class_simulated là một array, thứ tự khớp với class ID 0, 1, 2...
    # Số phần tử của nó phải bằng nc (số lớp trong YAML)
    # Giá trị mAP sẽ là ngẫu nhiên hoặc 0 nếu lớp đó không có GT trong tập test thực tế
    num_classes_in_yaml = len(class_names_from_yaml)
    maps_per_class_simulated = np.random.rand(num_classes_in_yaml) * 0.1 # mAP thấp giả lập
    
    # Giả sử DRONE (ID 2) và HELICOPTER (ID 3) có dữ liệu và mAP cao hơn
    if 'DRONE' in class_names_from_yaml:
        drone_idx = class_names_from_yaml.index('DRONE')
        maps_per_class_simulated[drone_idx] = 0.85 + random.uniform(-0.05, 0.05) # Giả lập mAP cao
    if 'HELICOPTER' in class_names_from_yaml:
        helicopter_idx = class_names_from_yaml.index('HELICOPTER')
        maps_per_class_simulated[helicopter_idx] = 0.70 + random.uniform(-0.05, 0.05) # Giả lập mAP khá

    print(f"  maps_per_class (giả lập): {maps_per_class_simulated}")
    print(f"  Thứ tự lớp tương ứng với maps_per_class ở trên (từ YAML): {class_names_from_yaml}")


    # 5. In ra cách các chỉ số sẽ được báo cáo
    print("\n--- Cách các chỉ số mAP theo lớp sẽ được báo cáo (DỰ KIẾN) ---")
    # Logic này giống với trong script train của bạn
    
    class_map_dict_check = {}
    if maps_per_class_simulated is not None and len(maps_per_class_simulated) > 0:
        print("mAP@0.50-0.95 per class (dựa trên `class_names_from_yaml` và `maps_per_class_simulated`):")
        for i, class_name_in_yaml_check in enumerate(class_names_from_yaml):
            if i < len(maps_per_class_simulated):
                class_map_dict_check[class_name_in_yaml_check] = maps_per_class_simulated[i]
                print(f"  - {class_name_in_yaml_check} (ID {i} trong YAML): {maps_per_class_simulated[i]:.4f}")
            else:
                print(f"  - {class_name_in_yaml_check} (ID {i} trong YAML): N/A (index out of bounds for maps_per_class_simulated)")
    
    print("\nBáo cáo cho các lớp trong `cfg.MASTER_CLASS_LIST_FUSION`:")
    # cfg_check.MASTER_CLASS_LIST_FUSION nên là ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER', 'BACKGROUND']
    master_classes = cfg_check.MASTER_CLASS_LIST_FUSION
    if not master_classes:
        print("LỖI: cfg.MASTER_CLASS_LIST_FUSION không được định nghĩa trong config.")
        return
        
    for master_class_name_check in master_classes:
        if master_class_name_check in class_map_dict_check:
            # Lớp này có trong YAML và có giá trị mAP
            print(f"  - {master_class_name_check}: {class_map_dict_check[master_class_name_check]:.4f}")
        else:
            # Lớp này không có trong YAML (ví dụ: BACKGROUND), hoặc không có dữ liệu trong test (dẫn đến mAP=0 nếu có trong YAML)
            if master_class_name_check == "BACKGROUND":
                print(f"  - {master_class_name_check}: N/A (YOLO detection mAP không áp dụng trực tiếp cho lớp 'BACKGROUND' nếu nó không được annotate như một đối tượng)")
            elif master_class_name_check in class_names_from_yaml: # Có trong YAML nhưng không có trong class_map_dict_check (ví dụ mAP=0 hoặc lỗi)
                # Tìm index của nó trong class_names_from_yaml để lấy giá trị mAP tương ứng nếu có
                try:
                    idx_in_yaml = class_names_from_yaml.index(master_class_name_check)
                    if idx_in_yaml < len(maps_per_class_simulated):
                         print(f"  - {master_class_name_check}: {maps_per_class_simulated[idx_in_yaml]:.4f} (Có trong YAML, mAP từ simulated array)")
                    else:
                         print(f"  - {master_class_name_check}: 0.0000 (Có trong YAML, nhưng index maps_per_class không đủ)")
                except ValueError: # Không thực sự có trong class_names_from_yaml (trường hợp này không nên xảy ra nếu logic trên đúng)
                     print(f"  - {master_class_name_check}: N/A (Không có trong 'names' của YAML)")
            else: # Không có trong YAML
                print(f"  - {master_class_name_check}: N/A (Không có trong 'names' của YAML được dùng để huấn luyện/đánh giá YOLO)")


    print("\n===== VCAM EVALUATION REPORTING CHECK FINISHED =====")
    print("LƯU Ý: Đây là MÔ PHỎNG cách báo cáo. Kết quả mAP thực tế sẽ phụ thuộc vào dữ liệu ground truth trong tập test của bạn.")

if __name__ == '__main__':
    check_vcam_evaluation_reporting_logic()