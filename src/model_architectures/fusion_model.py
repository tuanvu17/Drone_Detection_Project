# Drone_Detection_Project/src/model_architectures/fusion_model.py
import os # Thêm os để kiểm tra file nếu cần trong __main__
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, Add, Multiply,
    LayerNormalization, Activation, GlobalAveragePooling1D
)
from tensorflow.keras import backend as K
import pickle


# ============================================================================
# CUSTOM FUSION LAYERS
# ============================================================================

class AttentionFusionLayer(tf.keras.layers.Layer):
    """
    Attention-based Fusion: Học trọng số attention cho từng modality.
    Cho phép mô hình tự động xác định modality nào quan trọng hơn.
    """
    def __init__(self, units=64, **kwargs):
        super(AttentionFusionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape là list của 2 shapes: [audio_shape, vcam_shape]
        self.W_audio = self.add_weight(
            name='W_audio',
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_vcam = self.add_weight(
            name='W_vcam',
            shape=(input_shape[1][-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.V = self.add_weight(
            name='V',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionFusionLayer, self).build(input_shape)

    def call(self, inputs):
        audio_features, vcam_features = inputs

        # Project features to same space
        audio_proj = tf.tanh(tf.matmul(audio_features, self.W_audio))
        vcam_proj = tf.tanh(tf.matmul(vcam_features, self.W_vcam))

        # Compute attention scores
        audio_score = tf.matmul(audio_proj, self.V)
        vcam_score = tf.matmul(vcam_proj, self.V)

        # Stack and apply softmax
        scores = tf.concat([audio_score, vcam_score], axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum
        audio_weight = attention_weights[:, 0:1]
        vcam_weight = attention_weights[:, 1:2]

        # Cần project về cùng dimension trước khi weighted sum
        common_dim = max(audio_features.shape[-1], vcam_features.shape[-1])
        audio_dense = Dense(common_dim, use_bias=False)(audio_features)
        vcam_dense = Dense(common_dim, use_bias=False)(vcam_features)

        fused = audio_weight * audio_dense + vcam_weight * vcam_dense
        return fused

    def get_config(self):
        config = super(AttentionFusionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class GatedFusionLayer(tf.keras.layers.Layer):
    """
    Gated Fusion: Sử dụng cơ chế gate để điều khiển luồng thông tin.
    Tương tự như gate trong LSTM/GRU.
    """
    def __init__(self, output_dim=256, **kwargs):
        super(GatedFusionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        audio_dim = input_shape[0][-1]
        vcam_dim = input_shape[1][-1]

        # Gate weights
        self.W_gate = self.add_weight(
            name='W_gate',
            shape=(audio_dim + vcam_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_gate = self.add_weight(
            name='b_gate',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )

        # Transform weights for audio
        self.W_audio = self.add_weight(
            name='W_audio',
            shape=(audio_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        # Transform weights for vcam
        self.W_vcam = self.add_weight(
            name='W_vcam',
            shape=(vcam_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        super(GatedFusionLayer, self).build(input_shape)

    def call(self, inputs):
        audio_features, vcam_features = inputs

        # Concatenate for gate computation
        concat_features = tf.concat([audio_features, vcam_features], axis=-1)

        # Compute gate (sigmoid activation)
        gate = tf.sigmoid(tf.matmul(concat_features, self.W_gate) + self.b_gate)

        # Transform features
        audio_transformed = tf.matmul(audio_features, self.W_audio)
        vcam_transformed = tf.matmul(vcam_features, self.W_vcam)

        # Gated fusion: gate controls how much of each modality to use
        fused = gate * audio_transformed + (1 - gate) * vcam_transformed
        return fused

    def get_config(self):
        config = super(GatedFusionLayer, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


class BilinearFusionLayer(tf.keras.layers.Layer):
    """
    Bilinear Fusion: Mô hình hóa tương tác bậc hai giữa hai modality.
    output = x1^T * W * x2 + b
    """
    def __init__(self, output_dim=256, **kwargs):
        super(BilinearFusionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        audio_dim = input_shape[0][-1]
        vcam_dim = input_shape[1][-1]

        # Bilinear weight tensor
        self.W = self.add_weight(
            name='bilinear_W',
            shape=(audio_dim, vcam_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='bilinear_b',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )

        super(BilinearFusionLayer, self).build(input_shape)

    def call(self, inputs):
        audio_features, vcam_features = inputs

        # Bilinear interaction: sum over i,j of audio[i] * W[i,j,k] * vcam[j]
        # Reshape for efficient computation
        # audio: (batch, audio_dim) -> (batch, audio_dim, 1)
        # vcam: (batch, vcam_dim) -> (batch, 1, vcam_dim)

        batch_size = tf.shape(audio_features)[0]

        # Compute bilinear product
        # First: audio @ W -> (batch, vcam_dim, output_dim)
        audio_W = tf.einsum('bi,ijk->bjk', audio_features, self.W)
        # Then: (audio @ W) * vcam -> (batch, output_dim)
        bilinear_out = tf.einsum('bjk,bj->bk', audio_W, vcam_features)

        return bilinear_out + self.b

    def get_config(self):
        config = super(BilinearFusionLayer, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


class MultiHeadCrossAttentionFusion(tf.keras.layers.Layer):
    """
    Multi-Head Cross-Attention Fusion:
    Sử dụng cross-attention giữa hai modality với nhiều attention heads.
    """
    def __init__(self, num_heads=4, key_dim=64, output_dim=256, **kwargs):
        super(MultiHeadCrossAttentionFusion, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.mha_audio_to_vcam = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name='mha_audio_to_vcam'
        )
        self.mha_vcam_to_audio = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name='mha_vcam_to_audio'
        )
        self.layer_norm = LayerNormalization()
        self.output_dense = Dense(self.output_dim, activation='relu')

        super(MultiHeadCrossAttentionFusion, self).build(input_shape)

    def call(self, inputs):
        audio_features, vcam_features = inputs

        # Expand dims for attention (add sequence dimension)
        audio_expanded = tf.expand_dims(audio_features, axis=1)  # (batch, 1, audio_dim)
        vcam_expanded = tf.expand_dims(vcam_features, axis=1)    # (batch, 1, vcam_dim)

        # Cross attention: audio attends to vcam
        audio_attended = self.mha_audio_to_vcam(
            query=audio_expanded,
            key=vcam_expanded,
            value=vcam_expanded
        )

        # Cross attention: vcam attends to audio
        vcam_attended = self.mha_vcam_to_audio(
            query=vcam_expanded,
            key=audio_expanded,
            value=audio_expanded
        )

        # Squeeze and concatenate
        audio_attended = tf.squeeze(audio_attended, axis=1)
        vcam_attended = tf.squeeze(vcam_attended, axis=1)

        # Combine attended features
        combined = tf.concat([audio_attended, vcam_attended], axis=-1)
        combined = self.layer_norm(combined)

        return self.output_dense(combined)

    def get_config(self):
        config = super(MultiHeadCrossAttentionFusion, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'output_dim': self.output_dim
        })
        return config
# Thử import cfg, nếu không được thì sẽ dùng giá trị mặc định trong __main__ (để file này vẫn có thể chạy riêng để test build)
try:
    from src.config_loader.loader import get_config
    cfg_module_level = get_config() # Tải config ở mức module để có thể dùng trong __main__
except ImportError:
    cfg_module_level = None
    print("Cảnh báo (fusion_model.py): Không thể tải config ở mức module. Phần __main__ có thể cần giá trị mặc định.")


# ============================================================================
# FUSION METHOD ENUM
# ============================================================================

class FusionMethod:
    """Enum cho các phương pháp fusion có sẵn."""
    CONCATENATE = 'concatenate'           # Nối đặc trưng (phương pháp gốc)
    ADD = 'add'                           # Cộng đặc trưng (yêu cầu cùng dimension)
    MULTIPLY = 'multiply'                 # Nhân element-wise (yêu cầu cùng dimension)
    ATTENTION = 'attention'               # Attention-based fusion
    GATED = 'gated'                       # Gated fusion (như LSTM gate)
    BILINEAR = 'bilinear'                 # Bilinear interaction
    CROSS_ATTENTION = 'cross_attention'   # Multi-head cross-attention


def build_early_fusion_model(audio_input_shape,
                             vcam_feature_input_shape,
                             num_fusion_classes,
                             audio_pretrained_model_path,
                             fusion_learning_rate,
                             audio_branch_trainable=False,
                             audio_feature_layer_name=cfg_module_level.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION,
                            #  fusion_method=FusionMethod.CONCATENATE,
                            #  fusion_method=FusionMethod.ATTENTION,
                             fusion_method=FusionMethod.GATED,
                             fusion_output_dim=256):
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
        fusion_method (str): Phương pháp fusion để kết hợp đặc trưng. Các giá trị:
                            - 'concatenate': Nối đặc trưng (mặc định)
                            - 'add': Cộng đặc trưng
                            - 'multiply': Nhân element-wise
                            - 'attention': Attention-based fusion
                            - 'gated': Gated fusion
                            - 'bilinear': Bilinear fusion
                            - 'cross_attention': Multi-head cross-attention
        fusion_output_dim (int): Dimension đầu ra cho các phương pháp fusion (mặc định: 256).

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
    print(f"Sử dụng phương pháp fusion: {fusion_method}")

    if fusion_method == FusionMethod.CONCATENATE:
        # Phương pháp gốc: Nối đặc trưng
        merged_features = Concatenate(name='concatenate_audio_vcam')([audio_extracted_features, x_vcam])

    elif fusion_method == FusionMethod.ADD:
        # Cộng đặc trưng (cần project về cùng dimension)
        audio_proj = Dense(fusion_output_dim, activation='relu', name='audio_proj_for_add')(audio_extracted_features)
        vcam_proj = Dense(fusion_output_dim, activation='relu', name='vcam_proj_for_add')(x_vcam)
        merged_features = Add(name='add_audio_vcam')([audio_proj, vcam_proj])

    elif fusion_method == FusionMethod.MULTIPLY:
        # Nhân element-wise (cần project về cùng dimension)
        audio_proj = Dense(fusion_output_dim, activation='relu', name='audio_proj_for_multiply')(audio_extracted_features)
        vcam_proj = Dense(fusion_output_dim, activation='relu', name='vcam_proj_for_multiply')(x_vcam)
        merged_features = Multiply(name='multiply_audio_vcam')([audio_proj, vcam_proj])

    elif fusion_method == FusionMethod.ATTENTION:
        # Attention-based fusion
        merged_features = AttentionFusionLayer(units=64, name='attention_fusion')([audio_extracted_features, x_vcam])

    elif fusion_method == FusionMethod.GATED:
        # Gated fusion
        merged_features = GatedFusionLayer(output_dim=fusion_output_dim, name='gated_fusion')([audio_extracted_features, x_vcam])

    elif fusion_method == FusionMethod.BILINEAR:
        # Bilinear fusion
        merged_features = BilinearFusionLayer(output_dim=fusion_output_dim, name='bilinear_fusion')([audio_extracted_features, x_vcam])

    elif fusion_method == FusionMethod.CROSS_ATTENTION:
        # Multi-head cross-attention fusion
        merged_features = MultiHeadCrossAttentionFusion(
            num_heads=4,
            key_dim=64,
            output_dim=fusion_output_dim,
            name='cross_attention_fusion'
        )([audio_extracted_features, x_vcam])

    else:
        valid_methods = [m.name for m in FusionMethod]
        raise ValueError(
            f"Phương pháp fusion '{fusion_method}' không được hỗ trợ. "
            f"Các phương pháp hợp lệ: {valid_methods}")

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
                         name=f'vcam_audio_{fusion_method}_fusion_model')

    fusion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fusion_learning_rate),
                         loss=loss_function,
                         metrics=['accuracy'])

    print(f"--- Early Fusion Model Built Successfully (method: {fusion_method}) ---")
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
                # Test tất cả các phương pháp fusion
                fusion_methods_to_test = [
                    FusionMethod.CONCATENATE,
                    FusionMethod.ADD,
                    FusionMethod.MULTIPLY,
                    FusionMethod.ATTENTION,
                    FusionMethod.GATED,
                    FusionMethod.BILINEAR,
                    FusionMethod.CROSS_ATTENTION,
                ]

                for method in fusion_methods_to_test:
                    print(f"\n{'='*80}")
                    print(f"TESTING FUSION METHOD: {method}")
                    print(f"{'='*80}")
                    try:
                        test_model = build_early_fusion_model(
                            audio_input_shape=audio_shape_test,
                            vcam_feature_input_shape=vcam_shape_test,
                            num_fusion_classes=num_classes_test,
                            audio_pretrained_model_path=audio_model_path_test,
                            fusion_learning_rate=lr_test,
                            audio_branch_trainable=False,
                            audio_feature_layer_name=cfg_test.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION,
                            fusion_method=method,
                            fusion_output_dim=256
                        )
                        print(f"\n--- Tóm tắt Kiến trúc Mô hình Fusion ({method}) ---")
                        test_model.summary(line_length=150)
                        print(f"\n[OK] Kiểm thử {method} hoàn tất thành công!")

                        # Clear session để tránh memory leak khi test nhiều models
                        tf.keras.backend.clear_session()

                    except Exception as e_build_test:
                        print(f"[FAILED] Lỗi khi test {method}: {e_build_test}")
                        import traceback
                        traceback.print_exc()

                print(f"\n{'='*80}")
                print("HOÀN TẤT KIỂM THỬ TẤT CẢ CÁC PHƯƠNG PHÁP FUSION")
                print(f"{'='*80}")