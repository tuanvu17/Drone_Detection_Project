# Drone_Detection_Project/src/model_architectures/fusion_model.py
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, LayerNormalization
)
import pickle


# ============================================================================
# CUSTOM FUSION LAYERS (FIXED VERSION)
# ============================================================================

class AttentionFusionLayer(tf.keras.layers.Layer):
    """
    Attention-based Fusion: Học trọng số attention cho từng modality.
    """
    def __init__(self, units=64, **kwargs):
        super(AttentionFusionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # input_shape là list của 2 shapes: [audio_shape, vcam_shape]
        audio_dim = input_shape[0][-1]
        vcam_dim = input_shape[1][-1]
        
        # Attention weights
        self.W_audio = self.add_weight(
            name='W_audio',
            shape=(audio_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_vcam = self.add_weight(
            name='W_vcam',
            shape=(vcam_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.V = self.add_weight(
            name='V',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Common dimension là max của hai dimensions
        self.common_dim = max(audio_dim, vcam_dim)
        
        # Tạo layers trong build() thay vì call()
        self.audio_dense = Dense(
            self.common_dim, 
            use_bias=False,
            name='attention_audio_projection'
        )
        self.vcam_dense = Dense(
            self.common_dim, 
            use_bias=False,
            name='attention_vcam_projection'
        )
        
        # Build các dense layers
        self.audio_dense.build((None, audio_dim))
        self.vcam_dense.build((None, vcam_dim))
        
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
        
        # Project về cùng dimension - sử dụng layers đã tạo trong build()
        audio_dense = self.audio_dense(audio_features)
        vcam_dense = self.vcam_dense(vcam_features)
        
        # Weighted sum
        fused = audio_weight * audio_dense + vcam_weight * vcam_dense
        return fused
        
    def get_config(self):
        config = super(AttentionFusionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class GatedFusionLayer(tf.keras.layers.Layer):
    """
    Gated Fusion: Sử dụng cơ chế gate để điều khiển luồng thông tin.
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
        
        # Gated fusion
        fused = gate * audio_transformed + (1 - gate) * vcam_transformed
        return fused
        
    def get_config(self):
        config = super(GatedFusionLayer, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


class HybridFusionLayer(tf.keras.layers.Layer):
    """
    Hybrid Fusion: Kết hợp Attention và Gated Fusion.
    """
    def __init__(self, output_dim=256, **kwargs):
        super(HybridFusionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        
    def build(self, input_shape):
        audio_dim = input_shape[0][-1]
        vcam_dim = input_shape[1][-1]
        
        # Attention weights
        self.W_att_audio = self.add_weight(
            name='W_att_audio',
            shape=(audio_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_att_vcam = self.add_weight(
            name='W_att_vcam',
            shape=(vcam_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.V_att = self.add_weight(
            name='V_att',
            shape=(self.output_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Common dimension cho attention fusion
        self.common_dim = max(audio_dim, vcam_dim)
        
        # Attention projection layers
        self.attention_audio_proj = Dense(
            self.common_dim,
            use_bias=False,
            name='hybrid_attention_audio_proj'
        )
        self.attention_vcam_proj = Dense(
            self.common_dim,
            use_bias=False,
            name='hybrid_attention_vcam_proj'
        )
        
        # Build attention projection layers
        self.attention_audio_proj.build((None, audio_dim))
        self.attention_vcam_proj.build((None, vcam_dim))
        
        # Gated fusion weights
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
        
        # Feature transformation weights
        self.W_audio = self.add_weight(
            name='W_audio',
            shape=(audio_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_vcam = self.add_weight(
            name='W_vcam',
            shape=(vcam_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Output projection
        self.W_out = self.add_weight(
            name='W_out',
            shape=(2 * self.output_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_out = self.add_weight(
            name='b_out',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Attention output projection (chỉ cần nếu common_dim != output_dim)
        if self.common_dim != self.output_dim:
            self.attention_output_proj = Dense(
                self.output_dim,
                use_bias=False,
                name='hybrid_attention_output_proj'
            )
            self.attention_output_proj.build((None, self.common_dim))
        else:
            self.attention_output_proj = None
        
        super(HybridFusionLayer, self).build(input_shape)
        
    def call(self, inputs):
        audio_features, vcam_features = inputs
        
        # --- Bước 1: Attention Fusion ---
        audio_proj = tf.tanh(tf.matmul(audio_features, self.W_att_audio))
        vcam_proj = tf.tanh(tf.matmul(vcam_features, self.W_att_vcam))
        
        audio_score = tf.matmul(audio_proj, self.V_att)
        vcam_score = tf.matmul(vcam_proj, self.V_att)
        
        scores = tf.concat([audio_score, vcam_score], axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        audio_weight = attention_weights[:, 0:1]
        vcam_weight = attention_weights[:, 1:2]
        
        # Project features to common dimension
        audio_dense = self.attention_audio_proj(audio_features)
        vcam_dense = self.attention_vcam_proj(vcam_features)
        
        attention_fused = audio_weight * audio_dense + vcam_weight * vcam_dense
        
        # Project to output_dim if needed
        if self.attention_output_proj is not None:
            attention_fused = self.attention_output_proj(attention_fused)
        
        # --- Bước 2: Gated Fusion ---
        concat_features = tf.concat([audio_features, vcam_features], axis=-1)
        gate = tf.sigmoid(tf.matmul(concat_features, self.W_gate) + self.b_gate)
        
        audio_transformed = tf.matmul(audio_features, self.W_audio)
        vcam_transformed = tf.matmul(vcam_features, self.W_vcam)
        
        gated_fused = gate * audio_transformed + (1 - gate) * vcam_transformed
        
        # --- Bước 3: Kết hợp cả hai ---
        combined = tf.concat([attention_fused, gated_fused], axis=-1)
        
        # Final projection
        fused = tf.matmul(combined, self.W_out) + self.b_out
        fused = tf.nn.relu(fused)
        
        return fused
        
    def get_config(self):
        config = super(HybridFusionLayer, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


# ============================================================================
# FUSION METHOD ENUM
# ============================================================================

class FusionMethod:
    """Enum cho các phương pháp fusion."""
    CONCATENATE = 'concatenate'     # Nối đặc trưng
    ATTENTION = 'attention'         # Attention-based fusion
    GATED = 'gated'                 # Gated fusion
    HYBRID = 'hybrid'               # Hybrid fusion (Attention + Gated)


# ============================================================================
# BUILD FUSION MODEL
# ============================================================================

def build_early_fusion_model(audio_input_shape,
                             vcam_feature_input_shape,
                             num_fusion_classes,
                             audio_pretrained_model_path,
                             fusion_learning_rate,
                             audio_branch_trainable=False,
                             audio_feature_layer_name=None,
                             fusion_method=FusionMethod.CONCATENATE,
                            #  fusion_method=FusionMethod.GATED,
                            #  fusion_method=FusionMethod.ATTENTION,
                            #  fusion_method=FusionMethod.HYBRID,
                             fusion_output_dim=256):
    """
    Xây dựng mô hình Early Fusion kết hợp đặc trưng âm thanh và VCam.
    """
    print("--- Building Early Fusion Model (VCam Features + Audio MFCCs) ---")
    
    # --- 1. Định nghĩa Đầu vào ---
    audio_input = Input(shape=audio_input_shape, name='audio_mfcc_input')
    vcam_feature_input = Input(shape=vcam_feature_input_shape, name='vcam_extracted_feature_input')
    
    # --- 2. Xây dựng Nhánh Âm thanh ---
    try:
        audio_full_pretrained_model = load_model(audio_pretrained_model_path)
        print(f"Đã tải mô hình audio pre-trained từ: {audio_pretrained_model_path}")
        
        # Build model nếu chưa build
        if not audio_full_pretrained_model.built:
            print(f"Building mô hình với input_shape: {(None, *audio_input_shape)}")
            audio_full_pretrained_model.build(input_shape=(None, *audio_input_shape))
    except Exception as e:
        print(f"LỖI khi tải mô hình audio: {e}")
        raise
    
    # Lấy lớp trích xuất đặc trưng
    if audio_feature_layer_name is None:
        # Nếu không chỉ định, lấy lớp áp chót
        audio_feature_layer_name = audio_full_pretrained_model.layers[-2].name
        print(f"Tự động chọn lớp trích xuất đặc trưng: {audio_feature_layer_name}")
    
    try:
        audio_feature_extractor_layer = audio_full_pretrained_model.get_layer(audio_feature_layer_name)
    except ValueError:
        print(f"LỖI: Không tìm thấy lớp '{audio_feature_layer_name}'")
        audio_full_pretrained_model.summary()
        raise
    
    # Tạo sub-model trích xuất đặc trưng
    audio_feature_extractor_model = Model(
        inputs=audio_full_pretrained_model.inputs,
        outputs=audio_feature_extractor_layer.output,
        name='audio_feature_extractor'
    )
    
    audio_feature_extractor_model.trainable = audio_branch_trainable
    print(f"Nhánh Audio trainable = {audio_branch_trainable}")
    
    # Trích xuất đặc trưng audio
    audio_extracted_features = audio_feature_extractor_model(audio_input)
    
    # Flatten nếu cần (tùy thuộc vào shape của audio features)
    if len(audio_extracted_features.shape) > 2:
        audio_extracted_features = tf.keras.layers.Flatten()(audio_extracted_features)
        print(f"Đã flatten audio features: {audio_extracted_features.shape}")
    
    # --- 3. Xây dựng Nhánh VCam ---
    x_vcam = Dense(128, activation='relu', name='vcam_fusion_dense_1')(vcam_feature_input)
    x_vcam = Dropout(0.3, name='vcam_fusion_dropout_1')(x_vcam)
    
    # --- 4. Kết hợp (Fusion) ---
    print(f"Sử dụng phương pháp fusion: {fusion_method}")
    
    if fusion_method == FusionMethod.CONCATENATE:
        # Phương pháp gốc: Nối đặc trưng
        merged_features = Concatenate(name='concatenate_audio_vcam')([audio_extracted_features, x_vcam])
        
    elif fusion_method == FusionMethod.ATTENTION:
        # Attention-based fusion
        merged_features = AttentionFusionLayer(units=64, name='attention_fusion')(
            [audio_extracted_features, x_vcam]
        )
        
    elif fusion_method == FusionMethod.GATED:
        # Gated fusion
        merged_features = GatedFusionLayer(output_dim=fusion_output_dim, name='gated_fusion')(
            [audio_extracted_features, x_vcam]
        )
        
    elif fusion_method == FusionMethod.HYBRID:
        # Hybrid fusion
        merged_features = HybridFusionLayer(output_dim=fusion_output_dim, name='hybrid_fusion')(
            [audio_extracted_features, x_vcam]
        )
        
    else:
        valid_methods = [FusionMethod.CONCATENATE, FusionMethod.ATTENTION, 
                        FusionMethod.GATED, FusionMethod.HYBRID]
        raise ValueError(
            f"Phương pháp fusion '{fusion_method}' không được hỗ trợ. "
            f"Các phương pháp hợp lệ: {valid_methods}")
    
    # --- 5. Các lớp Phân loại ---
    x = Dense(256, activation='relu', name='fusion_dense_1')(merged_features)
    x = Dropout(0.3, name='fusion_dropout_1')(x)
    x = Dense(128, activation='relu', name='fusion_dense_2')(x)
    x = Dropout(0.3, name='fusion_dropout_2')(x)
    
    # Xác định hàm kích hoạt và loss
    if num_fusion_classes == 1:
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
    elif num_fusion_classes >= 2:
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'
    else:
        raise ValueError(f"num_fusion_classes ({num_fusion_classes}) không hợp lệ.")
    
    fusion_output = Dense(num_fusion_classes, activation=output_activation, name='fusion_output')(x)
    
    # --- 6. Tạo và Biên dịch Mô hình ---
    fusion_model = Model(
        inputs=[audio_input, vcam_feature_input],
        outputs=fusion_output,
        name=f'vcam_audio_{fusion_method}_fusion_model'
    )
    
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fusion_learning_rate),
        loss=loss_function,
        metrics=['accuracy']
    )
    
    print(f"--- Early Fusion Model Built Successfully (method: {fusion_method}) ---")
    return fusion_model
# ============================================================================
# END OF FILE