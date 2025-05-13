# Drone_Detection_Project/src/training/train_audio_pipeline.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# SỬA IMPORT: Sử dụng import tuyệt đối từ package src
from src.data_processing.audio_utils import (
    check_directory_structure,
    load_select_split_process_data,
    pad_features,
    scale_features
)
from src.model_architectures.audio_model import build_audio_model
from src.evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix
from src.evaluation.evaluate_audio import test_audio_model_on_saved_files


def run_audio_training_pipeline(
        base_path, classes, sample_rate, segment_duration,
        n_mfcc, n_fft, hop_length,
        validation_split, num_test_files_per_class,
        epochs, batch_size, learning_rate,
        # Đường dẫn lưu trữ
        model_save_path, label_encoder_path, scaler_path, max_len_path,
        test_files_list_path, training_history_plot_path,
        confusion_matrix_val_plot_path, confusion_matrix_test_plot_path, # Thêm _test_
        detailed_test_results_path,
        # Thêm các đường dẫn lưu kết quả test
        predictions_save_path, true_labels_save_path, class_names_save_path
    ):
    # ... (Nội dung hàm run_audio_training_pipeline như bạn cung cấp ở câu hỏi trước) ...
    # Đảm bảo nó gọi đúng các hàm đã import và truyền đúng các tham số đường dẫn.
    print("===== STARTING AUDIO TRAINING PIPELINE =====")

    # 1. Kiểm tra thư mục
    if not check_directory_structure(base_path, classes, num_test_files_per_class):
        print("Directory structure check failed. Exiting.")
        return

    # 2. Tải, chọn file test, xử lý data train/val
    print("Step 2: Loading and processing data...")
    train_val_features, encoded_train_val_labels, label_encoder, _ = \
        load_select_split_process_data(
            base_path, classes, segment_duration, sample_rate,
            n_mfcc, n_fft, hop_length,
            num_test_files_per_class, test_files_list_path
        )

    if not train_val_features: # Kiểm tra nếu không có features nào được trả về
        print("ERROR: No features loaded from data processing. Exiting pipeline.")
        return

    print(f"Saving LabelEncoder to {label_encoder_path}")
    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
    with open(label_encoder_path, 'wb') as f: pickle.dump(label_encoder, f)

    # 3. Đệm features train/val
    print("\nStep 3: Padding features...")
    features_padded, max_len = pad_features(train_val_features)
    if features_padded.size == 0:
        print("ERROR: No features to train on after padding. Exiting.")
        return
    print(f"Saving Max sequence length ({max_len}) to {max_len_path}")
    os.makedirs(os.path.dirname(max_len_path), exist_ok=True)
    with open(max_len_path, 'wb') as f: pickle.dump(max_len, f)

    # 4. Chia Train/Validation
    print("\nStep 4: Splitting train/validation data...")
    X_train, X_val, y_train, y_val = train_test_split(
        features_padded, encoded_train_val_labels,
        test_size=validation_split, random_state=42, stratify=encoded_train_val_labels
    )
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    if X_train.size > 0 : print(f"Train data shape: {X_train.shape}")
    if X_val.size > 0 : print(f"Validation data shape: {X_val.shape}")

    # 5. Chuẩn hóa dữ liệu
    print("\nStep 5: Scaling features...")
    X_train_scaled, scaler = scale_features(X_train)
    X_val_scaled, _ = scale_features(X_val, scaler=scaler)
    print(f"Saving StandardScaler to {scaler_path}")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)

    # 6. Xây dựng mô hình
    print("\nStep 6: Building model...")
    if max_len == 0 or n_mfcc == 0:
        print(f"Error: max_len ({max_len}) or n_mfcc ({n_mfcc}) is zero. Cannot build model.")
        return
    input_shape = (max_len, n_mfcc)
    num_classes = len(classes)
    model = build_audio_model(input_shape, num_classes, learning_rate)
    if model: model.summary() # In summary nếu model được tạo thành công

    # 7. Huấn luyện mô hình
    print("\nStep 7: Training model...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
    ]
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1
    )
    print("--- Training Finished ---")

    # 8. Vẽ biểu đồ huấn luyện và confusion matrix trên tập validation
    print("\nStep 8: Plotting training history and validation confusion matrix...")
    plot_training_history(history, filename=training_history_plot_path)

    print(f"Loading best model from {model_save_path} for validation set confusion matrix...")
    try:
        best_model_for_val_eval = tf.keras.models.load_model(model_save_path)
        val_pred_probs = best_model_for_val_eval.predict(X_val_scaled)
        val_pred_classes = np.argmax(val_pred_probs, axis=1)
        plot_custom_confusion_matrix(y_val, val_pred_classes, list(label_encoder.classes_), # Chuyển sang list
                                     filename=confusion_matrix_val_plot_path)
    except Exception as e:
         print(f"Error during validation set confusion matrix plotting: {e}")

    # 9. Test mô hình trên tập test đã lưu (ĐÃ DI CHUYỂN RA NGOÀI, GỌI TỪ main_train_audio.py)
    print("===== AUDIO TRAINING PIPELINE FINISHED (Testing will be run from main script if enabled) =====")