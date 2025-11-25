# Drone_Detection_Project/src/model_architectures/fusion_model.py
import os # Thêm os để kiểm tra file nếu cần trong __main__
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate # LSTM, Bidirectional không cần import trực tiếp ở đây nữa
                                                                    # vì chúng đã là một phần của audio_full_pretrained_model
import pickle
# Thử import cfg, nếu không được thì sẽ dùng giá trị mặc định trong __main__ (để file này vẫn có thể chạy riêng để test build)
try:
    from src.config_loader.loader import get_config
    cfg_module_level = get_config() # Tải config ở mức module để có thể dùng trong __main__
except ImportError:
    cfg_module_level = None
    print("Cảnh báo (fusion_model.py): Không thể tải config ở mức module. Phần __main__ có thể cần giá trị mặc định.")


def build_early_fusion_model(audio_input_shape,
                             vcam_feature_input_shape,
                             num_fusion_classes,
                             audio_pretrained_model_path,
                             fusion_learning_rate,
                             audio_branch_trainable=False,
                             audio_feature_layer_name=cfg_module_level.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION): # Thêm tham số cho tên lớp trích xuất
    """
    Xây dựng mô hình Early Fusion kết hợp đặc trưng âm thanh và VCam.

    Args:
        audio_input_shape (tuple): Shape của đầu vào MFCC cho nhánh audio (ví dụ: (max_len, n_mfcc)).
        vcam_feature_input_shape (tuple): Shape của đầu vào vector đặc trưng VCam (ví dụ: (256,)).
        num_fusion_classes (int): Số lượng lớp output cho mô hình fusion.
        audio_pretrained_model_path (str): Đường dẫn đến file .keras của mô hình audio đã huấn luyện.
        fusion_learning_rate (float): Learning rate cho việc biên dịch mô hình fusion.
        audio_branch_trainable (bool): True nếu muốn mở băng và fine-tune nhánh audio, False để đóng băng.
        audio_feature_layer_name (str): Tên của lớp trong mô hình audio pre-trained
                                         mà output của nó sẽ được dùng làm đặc trưng audio.

    Returns:
        tensorflow.keras.models.Model: Mô hình Early Fusion đã được biên dịch.
    """
    print("--- Building Early Fusion Model (VCam Features + Audio MFCCs) ---")

    # --- 1. Định nghĩa Đầu vào cho từng nhánh của mô hình FUSION MỚI ---
    # Đây là các placeholder sẽ nhận dữ liệu khi mô hình fusion được huấn luyện/dự đoán
    audio_input_for_fusion = Input(shape=audio_input_shape, name='audio_mfcc_input')
    vcam_feature_input = Input(shape=vcam_feature_input_shape, name='vcam_extracted_feature_input')

    # --- 2. Xây dựng Nhánh Âm thanh (sử dụng mô hình pre-trained) ---
    try:
        audio_full_pretrained_model = load_model(audio_pretrained_model_path)
        print(f"Đã tải mô hình audio pre-trained từ: {audio_pretrained_model_path}")

        # Build mô hình audio pre-trained nếu nó chưa được build
        # Điều này cần thiết để truy cập thuộc tính .inputs hoặc .input
        if not audio_full_pretrained_model.built:
            print(f"Mô hình audio pre-trained chưa được build. Đang build với input_shape: {(None, *audio_input_shape)}")
            try:
                # (None, *audio_input_shape) sẽ là ví dụ (None, 98, 13)
                audio_full_pretrained_model.build(input_shape=(None, *audio_input_shape))
                print("Mô hình audio pre-trained đã được build.")
            except Exception as e_build:
                print(f"LỖI khi build mô hình audio pre-trained: {e_build}")
                print("Summary của audio_full_pretrained_model TRƯỚC KHI BUILD (nếu có thể):")
                try: audio_full_pretrained_model.summary(line_length=120)
                except: print("  Không thể in summary.")
                raise

    except Exception as e:
        print(f"LỖI: Không thể tải hoặc build mô hình audio pre-trained từ '{audio_pretrained_model_path}': {e}")
        raise

    # Lấy lớp trích xuất đặc trưng audio từ mô hình đã build
    try:
        audio_feature_extractor_layer = audio_full_pretrained_model.get_layer(audio_feature_layer_name)
    except ValueError:
        print(f"LỖI: Không tìm thấy lớp '{audio_feature_layer_name}' trong mô hình audio pre-trained.")
        print("Vui lòng kiểm tra summary của mô hình audio và cập nhật tên lớp trích xuất đặc trưng.")
        audio_full_pretrained_model.summary(line_length=120)
        raise

    # Tạo một mô hình con (sub-model) để trích xuất đặc trưng audio.
    # Đầu vào của sub-model này là đầu vào của mô hình audio gốc.
    # Đầu ra của sub-model này là output của lớp đã chọn (audio_feature_extractor_layer).
    audio_feature_extractor_model = Model(inputs=audio_full_pretrained_model.inputs, # Sử dụng .inputs (số nhiều)
                                          outputs=audio_feature_extractor_layer.output,
                                          name='audio_feature_extractor')

    # Đặt trạng thái đóng băng/mở băng cho nhánh audio
    audio_feature_extractor_model.trainable = audio_branch_trainable
    print(f"Nhánh Audio (từ pre-trained model, trích xuất từ lớp '{audio_feature_layer_name}') được đặt trainable = {audio_branch_trainable}")

    # Kết nối input của fusion model với nhánh audio feature extractor
    audio_extracted_features = audio_feature_extractor_model(audio_input_for_fusion)


    # --- 3. Xây dựng Nhánh VCam (nhận vector đặc trưng đã trích xuất) ---
    x_vcam = Dense(128, activation='relu', name='vcam_fusion_dense_1')(vcam_feature_input)
    x_vcam = Dropout(0.3, name='vcam_fusion_dropout_1')(x_vcam)
    # Hoặc có thể không cần lớp Dense này nếu vector đặc trưng VCam đã tốt:
    # x_vcam = vcam_feature_input

    # --- 4. Kết hợp (Fusion) các Đặc trưng ---
    merged_features = Concatenate(name='concatenate_audio_vcam')([audio_extracted_features, x_vcam])

    # --- 5. Các lớp Phân loại Phía sau ---
    x = Dense(256, activation='relu', name='fusion_dense_combined_1')(merged_features)
    x = Dropout(0.3, name='fusion_dropout_combined_1')(x)
    x = Dense(128, activation='relu', name='fusion_dense_combined_2')(x)
    x = Dropout(0.3, name='fusion_dropout_combined_2')(x)

    # Xác định hàm kích hoạt và loss cho lớp output dựa trên num_fusion_classes
    if num_fusion_classes == 1:
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
    elif num_fusion_classes >= 2: # Bao gồm cả trường hợp binary_softmax nếu num_fusion_classes=2
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy' # Giả sử nhãn là số nguyên
    else:
        raise ValueError(f"num_fusion_classes ({num_fusion_classes}) không hợp lệ. Phải >= 1.")

    fusion_output = Dense( num_fusion_classes, activation=output_activation, name='fusion_output')(x)

    # --- 6. Tạo và Biên dịch Mô hình Fusion ---
    fusion_model = Model(inputs=[audio_input_for_fusion, vcam_feature_input],
                         outputs=fusion_output,
                         name='vcam_audio_early_fusion_model')

    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fusion_learning_rate),
                         loss=loss_function,
                         metrics=['accuracy'])

    print("--- Early Fusion Model Built Successfully ---")
    # fusion_model.summary(line_length=120) # Có thể gọi summary ở script huấn luyện chính
    return fusion_model


if __name__ == '__main__':
    # Phần này dùng để kiểm thử nhanh kiến trúc mô hình
    print("Chạy kiểm thử xây dựng mô hình Fusion (đây không phải là script huấn luyện)...")
    if cfg_module_level is None:
        print("Lỗi: Không thể tải cấu hình (cfg_module_level is None). Không thể chạy kiểm thử build.")
    else:
        cfg_test = cfg_module_level
        print(f"Sử dụng cấu hình từ: config.project_config")
        # Giả sử các giá trị từ config
        # Cần đảm bảo AUDIO_MAX_LEN_PATH đã được tạo và chứa giá trị từ pipeline audio
        max_len_path_test = cfg_test.AUDIO_MAX_LEN_PATH
        if not os.path.exists(max_len_path_test):
            print(f"Lỗi: File max_len audio không tồn tại tại: {max_len_path_test}")
            print("Vui lòng chạy pipeline huấn luyện audio để tạo file này trước.")
        else:
            with open(max_len_path_test, 'rb') as f:
                loaded_audio_max_len_test = pickle.load(f)

            audio_shape_test = (loaded_audio_max_len_test, cfg_test.AUDIO_N_MFCC)
            vcam_shape_test = (cfg_test.FUSION_VCAM_FEATURE_DIM,) # Ví dụ: (256,)
            num_classes_test = cfg_test.FUSION_NUM_CLASSES
            audio_model_path_test = cfg_test.AUDIO_MODEL_SAVE_PATH
            lr_test = 0.0001 # Learning rate ví dụ

            if not os.path.exists(audio_model_path_test):
                print(f"Lỗi: File mô hình audio pre-trained không tồn tại tại: {audio_model_path_test}")
            else:
                print(f"\nĐang tạo mô hình fusion mẫu với audio_input_shape={audio_shape_test}, vcam_feature_input_shape={vcam_shape_test}")
                try:
                    test_model = build_early_fusion_model(
                        audio_input_shape=audio_shape_test,
                        vcam_feature_input_shape=vcam_shape_test,
                        num_fusion_classes=num_classes_test,
                        audio_pretrained_model_path=audio_model_path_test,
                        fusion_learning_rate=lr_test,
                        audio_branch_trainable=False, # Ví dụ đóng băng khi test build
                        audio_feature_layer_name= cfg_test.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION # Sử dụng tên lớp bạn đã xác nhận
                    )
                    print("\n--- Tóm tắt Kiến trúc Mô hình Fusion Mẫu ---")
                    test_model.summary(line_length=150) # Tăng line_length để dễ đọc hơn
                    # tf.keras.utils.plot_model(test_model, to_file="fusion_model_architecture.png", show_shapes=True, dpi=96)
                    # print("Đã lưu kiến trúc mô hình vào fusion_model_architecture.png (nếu uncomment)")
                    print("\nKiểm thử xây dựng mô hình Fusion hoàn tất.")
                except Exception as e_build_test:
                    print(f"Lỗi trong quá trình kiểm thử build_early_fusion_model: {e_build_test}")
                    import traceback
                    traceback.print_exc()