# Drone_Detection_Project/src/model_architectures/audio_model.py
import tensorflow as tf # Giữ lại các import cần thiết cho định nghĩa model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_audio_model(input_shape, num_classes, learning_rate=0.001):
    # ... (code hàm build_audio_model như bạn đã cung cấp) ...
    print("--- Building BiLSTM Model for Audio ---")
    if len(input_shape) != 2 or None in input_shape or any(s <= 0 for s in input_shape if s is not None):
        raise ValueError(f"Invalid input shape for model: {input_shape}. Must be (timesteps > 0, features > 0).")

    model = Sequential([
        Input(shape=input_shape, name='audio_input_layer'),
        Bidirectional(LSTM(64, return_sequences=True, name='bilstm_1'), name='bidirectional_1'),
        Dropout(0.3, name='dropout_1'),
        Bidirectional(LSTM(64, return_sequences=False, name='bilstm_2'), name='bidirectional_2'),
        Dropout(0.3, name='dropout_2'),
        Dense(64, activation='relu', name='dense_1'),
        Dropout(0.3, name='dropout_3'),
        Dense(num_classes, activation='softmax', name='output_softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # print(model.summary(line_length=120)) # Có thể bỏ print summary ở đây, để script chính gọi
    # print("-" * 50)
    return model