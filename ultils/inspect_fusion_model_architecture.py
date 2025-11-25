# /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/utils/inspect_fusion_model_architecture.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import pickle

# Import config
try:
    from src.config_loader.loader import get_config
except ImportError:
    import sys
    current_dir_inspect_fusion = os.path.dirname(os.path.abspath(__file__))
    src_dir_inspect_fusion = os.path.dirname(current_dir_inspect_fusion)
    project_root_inspect_fusion = os.path.dirname(src_dir_inspect_fusion)
    if project_root_inspect_fusion not in sys.path: sys.path.insert(0, project_root_inspect_fusion)
    if src_dir_inspect_fusion not in sys.path: sys.path.insert(0, src_dir_inspect_fusion)
    from config_loader.loader import get_config

cfg_inspect_fusion = get_config()

def inspect_fusion_model(fusion_model_path,
                         audio_finetuned_model_path,
                         audio_feature_extraction_layer_name_in_audio_model,
                         vcam_input_feature_dim):
    if not os.path.exists(fusion_model_path):
        print(f"LỖI: Không tìm thấy file mô hình Fusion tại: {fusion_model_path}")
        return
    if not os.path.exists(audio_finetuned_model_path):
        print(f"LỖI: Không tìm thấy file mô hình Audio fine-tuned tại: {audio_finetuned_model_path}")
        return

    print(f"--- Đang tải mô hình Early Fusion từ: {fusion_model_path} ---")
    try:
        fusion_model = load_model(fusion_model_path)
        print("Đã tải mô hình Fusion thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình Fusion: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- === TÓM TẮT KIẾN TRÚC MÔ HÌNH EARLY FUSION === ---")
    fusion_model.summary(line_length=150)

    print("\n\n--- === PHÂN TÍCH CHI TIẾT CÁC THÀNH PHẦN TRONG FUSION MODEL (từ Model Object) === ---")

    # 1. Đầu vào của Fusion Model
    print("\n1. Đầu vào của Mô hình Fusion:")
    try:
        audio_mfcc_input_layer = fusion_model.get_layer('audio_mfcc_input')
        # Đối với InputLayer, shape của output tensor chính là shape của input nó định nghĩa
        print(f"  - Input Audio (Tên: '{audio_mfcc_input_layer.name}'): Shape={audio_mfcc_input_layer.output.shape}")
        print(f"    (Dự kiến cho đặc trưng MFCC đã pad & scale)")
    except Exception as e:
        print(f"  Lỗi khi lấy thông tin Input Audio: {e}")

    try:
        vcam_feature_input_layer = fusion_model.get_layer('vcam_extracted_feature_input')
        print(f"  - Input VCam (Tên: '{vcam_feature_input_layer.name}'): Shape={vcam_feature_input_layer.output.shape}")
        print(f"    (Dự kiến cho vector đặc trưng VCam {vcam_input_feature_dim} chiều đã trích xuất)")
    except Exception as e:
        print(f"  Lỗi khi lấy thông tin Input VCam: {e}")


    # 2. Nhánh Xử lý Âm thanh (bên trong Fusion)
    print("\n2. Nhánh Xử lý Âm thanh (bên trong Fusion Model):")
    audio_feature_output_dim = "Không xác định"
    try:
        audio_feature_extractor_submodel = fusion_model.get_layer('audio_feature_extractor')
        print(f"  - Tên Submodel/Lớp trích xuất Audio: '{audio_feature_extractor_submodel.name}' (Loại: {type(audio_feature_extractor_submodel).__name__})")
        print(f"  - Đầu vào của Submodel này (MFCC): Shape={audio_feature_extractor_submodel.input.shape if not isinstance(audio_feature_extractor_submodel.input, list) else [inp.shape for inp in audio_feature_extractor_submodel.input]}")
        print(f"  - Đầu ra của Submodel này (Vector Đặc trưng Audio Sâu): Shape={audio_feature_extractor_submodel.output.shape if not isinstance(audio_feature_extractor_submodel.output, list) else [out.shape for out in audio_feature_extractor_submodel.output]}")
        audio_feature_output_dim = audio_feature_extractor_submodel.output.shape[-1]
        print(f"    => Vector Đặc trưng Audio Sâu có: {audio_feature_output_dim} chiều")
        print(f"  - Submodel này được tạo từ mô hình '{audio_finetuned_model_path}', sử dụng output của lớp có tên cuối cùng là '{audio_feature_extraction_layer_name_in_audio_model}' trong mô hình đó.")
    except ValueError:
        print(f"  LỖI: Không tìm thấy submodel/lớp có tên 'audio_feature_extractor'.")


    # 3. Nhánh Xử lý VCam (bên trong Fusion Model)
    print("\n3. Nhánh Xử lý VCam (bên trong Fusion Model):")
    vcam_dense1_layer_output_dim = "Không xác định"
    try:
        vcam_dense1_layer = fusion_model.get_layer('vcam_fusion_dense_1')
        print(f"  - Lớp '{vcam_dense1_layer.name}': Units={vcam_dense1_layer.units}, Activation='{vcam_dense1_layer.activation.__name__}'")
        print(f"    Output shape: {vcam_dense1_layer.output.shape}")
        vcam_dense1_layer_output_dim = vcam_dense1_layer.output.shape[-1]
        print(f"    => Output VCam sau Dense 1 có: {vcam_dense1_layer_output_dim} nơ-ron/chiều")

        vcam_dropout1_layer = fusion_model.get_layer('vcam_fusion_dropout_1')
        print(f"  - Lớp '{vcam_dropout1_layer.name}': Rate={vcam_dropout1_layer.rate}")
        print(f"    Output shape: {vcam_dropout1_layer.output.shape}")
    except ValueError:
        print("  Không tìm thấy một trong các lớp 'vcam_fusion_dense_1' hoặc 'vcam_fusion_dropout_1'.")


    # 4. Lớp Kết hợp (Concatenate)
    print("\n4. Lớp Kết hợp (Concatenate):")
    try:
        concat_layer = fusion_model.get_layer('concatenate_audio_vcam')
        print(f"  - Lớp '{concat_layer.name}':")
        # concat_layer.input là một list các tensor
        print(f"    Input shapes: {[inp.shape for inp in concat_layer.input]}")
        print(f"    Output shape (Vector Đặc trưng Kết hợp): {concat_layer.output.shape}")
        concat_output_dim = concat_layer.output.shape[-1]
        print(f"    => Vector Đặc trưng Kết hợp có: {concat_output_dim} chiều")
        if audio_feature_output_dim != "Không xác định" and vcam_dense1_layer_output_dim != "Không xác định":
            expected_concat_dim = audio_feature_output_dim + vcam_dense1_layer_output_dim
            print(f"       (Kiểm tra: {audio_feature_output_dim} (Audio) + {vcam_dense1_layer_output_dim} (VCam) = {expected_concat_dim})")
    except ValueError:
        print("  LỖI: Không tìm thấy lớp 'concatenate_audio_vcam'.")


    # 5. Các Lớp Phân loại (Classification Head)
    print("\n5. Các Lớp Phân loại (Classification Head):")
    classification_layers = [
        'fusion_dense_combined_1', 'fusion_dropout_combined_1',
        'fusion_dense_combined_2', 'fusion_dropout_combined_2',
        'fusion_output'
    ]
    for layer_name in classification_layers:
        try:
            layer = fusion_model.get_layer(layer_name)
            if isinstance(layer, tf.keras.layers.Dense):
                print(f"  - Lớp '{layer.name}' (Dense): Units={layer.units}, Activation='{layer.activation.__name__}'")
            elif isinstance(layer, tf.keras.layers.Dropout):
                print(f"  - Lớp '{layer.name}' (Dropout): Rate={layer.rate}")
            print(f"    Output shape: {layer.output.shape}") # Sử dụng layer.output.shape
            if layer_name == 'fusion_output':
                print(f"    => Số lớp Output cuối cùng: {layer.output.shape[-1]}")
        except ValueError:
            print(f"  Không tìm thấy lớp '{layer_name}'.")

    print("\n===== INSPECTION FINISHED =====")

if __name__ == '__main__':
    fusion_model_to_inspect_path = cfg_inspect_fusion.BEST_FUSION_MODEL_SAVE_PATH
    audio_ft_model_to_inspect_path = cfg_inspect_fusion.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH

    audio_layer_for_features = cfg_inspect_fusion.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION
    vcam_input_dim = cfg_inspect_fusion.FUSION_VCAM_FEATURE_DIM

    print(f"Kiểm tra mô hình Fusion: {fusion_model_to_inspect_path}")
    print(f"Mô hình Audio Fine-tuned nguồn cho nhánh Audio: {audio_ft_model_to_inspect_path}")
    print(f"Lớp trong mô hình Audio nguồn được dùng để tạo submodel 'audio_feature_extractor': '{audio_layer_for_features}'")
    print(f"Chiều đặc trưng VCam đầu vào (InputLayer) cho Fusion: {vcam_input_dim}")

    inspect_fusion_model(
        fusion_model_path=fusion_model_to_inspect_path,
        audio_finetuned_model_path=audio_ft_model_to_inspect_path,
        audio_feature_extraction_layer_name_in_audio_model=audio_layer_for_features,
        vcam_input_feature_dim=vcam_input_dim
    )