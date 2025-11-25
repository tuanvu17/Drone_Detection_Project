# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/inspect_yolo_architecture.py
import os
from ultralytics import YOLO
import torch

# Import config để lấy đường dẫn model
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_inspect_yolo = os.path.dirname(os.path.abspath(__file__))
    src_dir_inspect_yolo = os.path.dirname(current_dir_inspect_yolo)
    project_root_inspect_yolo = os.path.dirname(src_dir_inspect_yolo)
    if project_root_inspect_yolo not in sys.path: sys.path.insert(0, project_root_inspect_yolo)
    if src_dir_inspect_yolo not in sys.path: sys.path.insert(0, src_dir_inspect_yolo)
    from config_loader.loader import get_config

cfg_inspect_yolo = get_config()

def print_yolo_model_architecture_details(model_path, example_input_shape=(1, 3, 640, 640)):
    """
    Tải một mô hình YOLO từ đường dẫn, in ra summary (nếu có dạng Keras-like)
    và duyệt qua các module con để hiển thị thông tin chi tiết hơn.
    Cũng thử đưa một input giả qua mô hình để xem shape output của từng lớp.
    """
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy file mô hình YOLO tại: {model_path}")
        return

    print(f"--- Đang tải mô hình YOLO từ: {model_path} ---")
    try:
        model = YOLO(model_path)
        # model.fuse() # Fuse layers (Conv2d + BatchNorm2d) for faster inference, có thể thay đổi cấu trúc một chút
        print("Đã tải mô hình YOLO thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLO: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- === TÓM TẮT KIẾN TRÚC MÔ HÌNH YOLO (TỪ ULTRALYTICS) === ---")
    # YOLO của Ultralytics có thể không có hàm summary() trực tiếp như Keras.
    # Chúng ta sẽ duyệt qua các module của nó.
    # `model.model` là đối tượng DetectionModel (hoặc tương tự) của PyTorch.
    # `model.model.model` là một nn.Sequential chứa các lớp chính.
    
    if hasattr(model, 'model') and hasattr(model.model, 'model') and isinstance(model.model.model, torch.nn.Sequential):
        yolo_sequential_layers = model.model.model
        print(f"Mô hình bao gồm {len(yolo_sequential_layers)} module chính trong nn.Sequential:")
        
        # Tạo một input giả để theo dõi shape qua các lớp
        # Kích thước input cần khớp với imgsz mà model được huấn luyện hoặc sẽ dùng
        # Ví dụ: (batch_size, channels, height, width)
        # example_input_shape được truyền vào hàm, ví dụ (1, 3, cfg_inspect_yolo.VCAM_YOLO_IMG_SIZE, cfg_inspect_yolo.VCAM_YOLO_IMG_SIZE)
        
        # Lấy kích thước ảnh từ config nếu có, nếu không dùng giá trị mặc định của hàm
        try:
            imgsz_h = cfg_inspect_yolo.VCAM_YOLO_IMG_SIZE # Giả sử imgsz là số正方形
            imgsz_w = cfg_inspect_yolo.VCAM_YOLO_IMG_SIZE
            if hasattr(cfg_inspect_yolo, 'VCAM_YOUTUBE_FINETUNE_IMG_SIZE') and model_path == cfg_inspect_yolo.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH:
                imgsz_h = cfg_inspect_yolo.VCAM_YOUTUBE_FINETUNE_IMG_SIZE
                imgsz_w = cfg_inspect_yolo.VCAM_YOUTUBE_FINETUNE_IMG_SIZE
            current_example_input_shape = (1, 3, imgsz_h, imgsz_w)
            print(f"Sử dụng example_input_shape: {current_example_input_shape} (batch, channels, height, width)")
        except AttributeError:
            print(f"Cảnh báo: Không tìm thấy VCAM_YOLO_IMG_SIZE trong config. Sử dụng example_input_shape mặc định: {example_input_shape}")
            current_example_input_shape = example_input_shape
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # Chuyển model sang device
        model.eval()   # Đặt model ở chế độ đánh giá
        
        # Tạo input giả trên đúng device
        dummy_input = torch.randn(current_example_input_shape).to(device)
        
        print(f"\n{'Index':<5} | {'Layer Type':<45} | {'Output Shape':<25} | {'Parameters':<12} | {'Input From':<10}")
        print("-" * 100)

        current_input_tensor = dummy_input
        # Lưu trữ output của các lớp để dùng cho các lớp Concat
        layer_outputs = {}

        for i, layer_module in enumerate(yolo_sequential_layers):
            layer_name = f"layer_{i}" # Tên tạm thời
            layer_type = type(layer_module).__name__
            
            # Lấy thông tin "from" (đầu vào từ lớp nào) nếu có (từ cấu trúc .yaml hoặc model definition)
            # Đây là phần khó vì model.model.model là nn.Sequential, không lưu trực tiếp thông tin "from"
            # Thông tin này thường có trong file định nghĩa kiến trúc .yaml mà YOLO sử dụng khi build model
            # hoặc trong thuộc tính `f` của từng module nếu model được parse từ yaml.
            input_from_indices = "N/A" # Mặc định
            if hasattr(layer_module, 'f') and layer_module.f is not None:
                if isinstance(layer_module.f, int):
                    input_from_indices = str(layer_module.f)
                elif isinstance(layer_module.f, list):
                    input_from_indices = ", ".join(map(str, layer_module.f))


            # Xử lý input cho các lớp đặc biệt như Concat
            if isinstance(layer_module, torch.nn.modules.conv.Conv2d) or \
               isinstance(layer_module, torch.nn.modules.pooling.MaxPool2d) or \
               isinstance(layer_module, torch.nn.modules.upsampling.Upsample) or \
               "ultralytics.nn.modules" in str(type(layer_module)): # Hầu hết các khối của yolo

                # Đối với Concat, nó cần nhiều input
                if "Concat" in layer_type:
                    if hasattr(layer_module, 'f') and isinstance(layer_module.f, list):
                        inputs_to_concat = []
                        for from_idx in layer_module.f:
                            actual_idx = i + from_idx if from_idx < 0 else from_idx
                            if actual_idx in layer_outputs:
                                inputs_to_concat.append(layer_outputs[actual_idx])
                            else:
                                print(f"Cảnh báo: Không tìm thấy output của lớp {actual_idx} cho Concat tại lớp {i}")
                                # Bỏ qua lớp này nếu không đủ input
                                print(f"{i:<5} | {layer_type:<45} | {'ERROR - MISSING INPUT':<25} | ...")
                                continue # Chuyển sang module tiếp theo
                        if not inputs_to_concat or len(inputs_to_concat) != len(layer_module.f):
                             current_output_tensor_shape_str = "ERROR - Concat Input Mismatch"
                        else:
                             current_input_tensor = inputs_to_concat # Đây là list các tensor
                             try:
                                current_output_tensor = layer_module(current_input_tensor)
                                current_output_tensor_shape_str = str(tuple(current_output_tensor.shape))
                             except Exception as e_inf:
                                print(f"Lỗi khi forward qua lớp {i} ({layer_type}): {e_inf}")
                                current_output_tensor_shape_str = "ERROR - Inference"
                                current_output_tensor = None # Đặt là None để không lưu vào layer_outputs
                    else:
                        current_output_tensor_shape_str = "ERROR - Concat 'f' undefined"
                        current_output_tensor = None
                else: # Các lớp thông thường khác
                    if current_input_tensor is None: # Nếu lớp trước đó lỗi
                        print(f"{i:<5} | {layer_type:<45} | {'ERROR - PREV LAYER FAILED':<25} | ...")
                        continue
                    try:
                        current_output_tensor = layer_module(current_input_tensor)
                        current_output_tensor_shape_str = str(tuple(current_output_tensor.shape))
                    except Exception as e_inf:
                        print(f"Lỗi khi forward qua lớp {i} ({layer_type}): {e_inf}")
                        current_output_tensor_shape_str = "ERROR - Inference"
                        current_output_tensor = None


                if current_output_tensor is not None:
                    layer_outputs[i] = current_output_tensor # Lưu output của lớp hiện tại
                    current_input_tensor = current_output_tensor # Output này là input cho lớp tiếp theo (trừ khi có Concat)
                else: # Nếu có lỗi inference, đặt current_input_tensor là None để các lớp sau biết
                    current_input_tensor = None


            else: # Các loại lớp khác không xử lý (ví dụ Detect head)
                current_output_tensor_shape_str = "N/A (Special Head Layer)"
                # Không cập nhật current_input_tensor, vì lớp Detect thường là cuối cùng của một nhánh

            num_params = sum(p.numel() for p in layer_module.parameters() if p.requires_grad)
            
            print(f"{i:<5} | {layer_type:<45} | {current_output_tensor_shape_str:<25} | {num_params:<12} | {input_from_indices:<10}")

        print("-" * 100)
        
        # In tổng số tham số từ model object của YOLO
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total params (from model object): {total_params:,}")
            print(f"Trainable params (from model object): {trainable_params:,}")
        except:
            pass

    else:
        print("Không thể truy cập model.model.model như một nn.Sequential.")
        print("Thử in cấu trúc model YOLO bằng cách khác (nếu hỗ trợ):")
        # Một số phiên bản YOLO có thể có cách in summary riêng
        # Ví dụ, thử gọi model.info() hoặc xem các thuộc tính của model
        if hasattr(model, 'info'):
            model.info(verbose=True) # In thông tin chi tiết nếu có
        else:
            print(type(model.model)) # In ra type của model.model để xem nó là gì

    print("\n===== YOLO MODEL ARCHITECTURE INSPECTION FINISHED =====")

if __name__ == '__main__':
    # Chọn mô hình bạn muốn kiểm tra
    # 1. Mô hình VCam gốc
    # vcam_model_to_inspect_path = cfg_inspect_yolo.VCAM_YOLO_BEST_MODEL_SAVE_PATH
    # print(f"Kiểm tra Mô hình VCam GỐC: {vcam_model_to_inspect_path}")
    # if os.path.exists(vcam_model_to_inspect_path):
    #     print_yolo_model_architecture_details(vcam_model_to_inspect_path,
    #                                           example_input_shape=(1, 3, cfg_inspect_yolo.VCAM_YOLO_IMG_SIZE, cfg_inspect_yolo.VCAM_YOLO_IMG_SIZE))
    # else:
    #     print(f"File không tồn tại: {vcam_model_to_inspect_path}")

    print("-" * 50)

    # 2. Mô hình VCam đã fine-tune trên YouTube
    vcam_ft_model_to_inspect_path = cfg_inspect_yolo.VCAM_YOUTUBE_FINETUNED_BEST_MODEL_SAVE_PATH
    print(f"Kiểm tra Mô hình VCam FINE-TUNED: {vcam_ft_model_to_inspect_path}")
    if os.path.exists(vcam_ft_model_to_inspect_path):
        print_yolo_model_architecture_details(vcam_ft_model_to_inspect_path,
                                              example_input_shape=(1, 3, cfg_inspect_yolo.VCAM_YOUTUBE_FINETUNE_IMG_SIZE, cfg_inspect_yolo.VCAM_YOUTUBE_FINETUNE_IMG_SIZE))
    else:
        print(f"File không tồn tại: {vcam_ft_model_to_inspect_path}")