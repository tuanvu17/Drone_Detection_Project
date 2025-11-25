# # Drone_Detection_Project/src/training/fine_tune_audio_youtube_pipeline.py
# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import librosa
# import pandas as pd

# # Import từ các module trong src
# try:
#     from src.config_loader.loader import get_config
#     from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
#     from src.evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix
# except ImportError:
#     import sys
#     current_dir_ft_audio_yt = os.path.dirname(os.path.abspath(__file__))
#     src_dir_ft_audio_yt = os.path.dirname(current_dir_ft_audio_yt)
#     project_root_ft_audio_yt = os.path.dirname(src_dir_ft_audio_yt)
#     if project_root_ft_audio_yt not in sys.path: sys.path.insert(0, project_root_ft_audio_yt)
#     if src_dir_ft_audio_yt not in sys.path: sys.path.insert(0, src_dir_ft_audio_yt)

#     from config_loader.loader import get_config
#     from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
#     from evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix

# cfg = get_config()

# def load_audio_youtube_segments_and_extract_mfcc(metadata_file_path, segments_base_dir):
#     """
#     Tải dữ liệu audio segment .wav, trích xuất MFCC và nhãn từ file metadata.
#     Trả về list MFCC, list nhãn text, và DataFrame metadata gốc cùng list các ID/index gốc.
#     """
#     print(f"--- Đang tải audio segments cho fine-tune từ metadata: {metadata_file_path} ---")
#     if not os.path.exists(metadata_file_path):
#         print(f"LỖI: File metadata audio YouTube không tồn tại: {metadata_file_path}")
#         return [], [], None, []
#     try:
#         metadata_df = pd.read_csv(metadata_file_path)
#         print(f"Đã đọc {len(metadata_df)} dòng từ metadata.")
#     except Exception as e:
#         print(f"Lỗi khi đọc file metadata audio YouTube: {e}")
#         return [], [], None, []

#     all_mfcc_features = []
#     all_labels_text = []
#     all_original_indices_or_ids = [] # Để theo dõi index gốc cho việc tách metadata của test set

#     for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Loading audio segments & extracting MFCC for YT fine-tune"):
#         try:
#             audio_segment_rel_path = str(row['audio_segment_file']).strip()
#             label = str(row['label']).strip()
#             audio_segment_abs_path = os.path.join(segments_base_dir, audio_segment_rel_path)

#             if not os.path.exists(audio_segment_abs_path):
#                 # print(f"Cảnh báo: Thiếu file audio segment {audio_segment_abs_path} cho dòng {index}. Bỏ qua.")
#                 continue

#             audio_seg_data, sr_loaded = librosa.load(audio_segment_abs_path, sr=cfg.AUDIO_SAMPLE_RATE)
#             if sr_loaded != cfg.AUDIO_SAMPLE_RATE:
#                 pass # librosa.load đã resample

#             # Ngưỡng tối thiểu cho độ dài audio segment
#             min_samples_for_segment = int(cfg.AUDIO_SEGMENT_DURATION * cfg.AUDIO_SAMPLE_RATE * 0.5)
#             if len(audio_seg_data) < min_samples_for_segment:
#                 # print(f"    Segment {audio_segment_rel_path} quá ngắn ({len(audio_seg_data)} samples). Bỏ qua.")
#                 continue

#             mfccs = extract_mfcc(audio_seg_data, cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC, cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH)
#             all_mfcc_features.append(mfccs)
#             all_labels_text.append(label)
#             all_original_indices_or_ids.append(index) # Lưu index gốc của dòng metadata
#         except Exception as e:
#             print(f"Lỗi khi xử lý audio segment {row.get('audio_segment_file', f'index {index}')}: {e}")
#             continue
    
#     if not all_labels_text:
#         print("Không tải được mẫu audio MFCC nào cho fine-tuning từ metadata.")
#         return [], [], None, []

#     print(f"Đã tải và trích xuất MFCC cho {len(all_labels_text)} audio segments từ dữ liệu YouTube.")
#     return all_mfcc_features, all_labels_text, metadata_df, all_original_indices_or_ids


# def run_audio_youtube_finetuning_pipeline():
#     print("===== STARTING AUDIO MODEL FINE-TUNING PIPELINE ON YOUTUBE DATA (WITH TEST IN-DOMAIN SPLIT FOR AUDIO) =====")
#     if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories):
#          cfg.ensure_output_directories()

#     # 1. Tải Dữ liệu Audio Segments (từ file .wav) và Trích xuất MFCC
#     X_mfcc_all_list, y_labels_text_all, full_original_metadata_df, original_indices_all = \
#         load_audio_youtube_segments_and_extract_mfcc(
#             metadata_file_path=cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE, # Metadata trỏ đến file .wav segment
#             segments_base_dir=cfg.AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR # Thư mục chứa file .wav segment
#         )
#     if not y_labels_text_all:
#         print("Không có dữ liệu audio từ YouTube để fine-tune. Dừng pipeline.")
#         return

#     # 2. Mã hóa Nhãn
#     audio_yt_label_encoder = LabelEncoder()
#     # Fit encoder trên AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE (nên là MASTER_CLASS_LIST_FUSION)
#     audio_yt_label_encoder.fit(cfg.AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE)
#     print(f"Audio YouTube Fine-tune LabelEncoder classes: {list(audio_yt_label_encoder.classes_)}")
#     try:
#         y_encoded_all = audio_yt_label_encoder.transform(y_labels_text_all)
#     except ValueError as e:
#         print(f"LỖI Mã hóa Nhãn cho Audio YouTube: {e}")
#         print(f"  Các nhãn duy nhất trong dữ liệu: {np.unique(y_labels_text_all)}")
#         print(f"  Các lớp LabelEncoder đã học: {list(audio_yt_label_encoder.classes_)}")
#         return
#     print(f"Đang lưu Audio YouTube LabelEncoder vào: {cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH}")
#     os.makedirs(os.path.dirname(cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH), exist_ok=True)
#     with open(cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH, 'wb') as f: pickle.dump(audio_yt_label_encoder, f)

#     # 3. Chia thành (Train+Validation) và Test In-Domain (cho riêng Audio)
#     print("\n--- Chia dữ liệu Audio YouTube thành (Train+Val) và Test In-Domain riêng cho Audio ---")
#     # Kiểm tra điều kiện chia
#     can_split_test_audio = len(y_encoded_all) >= 2 and \
#                            (len(y_encoded_all) * cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO >= 1 or cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO == 0) and \
#                            (len(y_encoded_all) * (1-cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO) >= 1 or (1-cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO) == 0) and \
#                            (len(np.unique(y_encoded_all)) >= 2 or cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO == 0 or (1-cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO) == 0)

#     X_mfcc_train_val_list = []
#     y_train_val_encoded = np.array([])
#     X_mfcc_test_id_list = []
#     y_test_id_encoded = np.array([])
#     indices_train_val_for_metadata = []

#     if can_split_test_audio and cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO > 0:
#         try:
#             X_mfcc_train_val_list, X_mfcc_test_id_list, \
#             y_train_val_encoded, y_test_id_encoded, \
#             indices_train_val_for_metadata, indices_test_id_for_metadata = train_test_split(
#                 X_mfcc_all_list, y_encoded_all, np.array(original_indices_all), # Chia cả chỉ số gốc
#                 test_size=cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO, # Sử dụng tỷ lệ của Fusion
#                 random_state=42,
#                 stratify=y_encoded_all
#             )
#             print(f"Số mẫu Test In-Domain cho Audio: {len(y_test_id_encoded)}")
#             # Lưu Test In-Domain features và labels (cho Audio)
#             os.makedirs(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, exist_ok=True)
#             # Đệm và Scale Test In-Domain Audio features TRƯỚC KHI LƯU
#             if X_mfcc_test_id_list:
#                 # Tính max_len từ toàn bộ dữ liệu (hoặc chỉ train+val) rồi mới pad test
#                 # Để nhất quán, max_len nên được tính từ tập train cuối cùng
#                 # Hiện tại, chúng ta sẽ pad test sau khi có max_len từ train
#                 pass # Sẽ pad và scale sau khi có scaler và max_len từ tập train
            
#             # Lưu metadata cho Test In-Domain Audio
#             if full_original_metadata_df is not None and len(indices_test_id_for_metadata) > 0:
#                 test_id_metadata_audio_df = full_original_metadata_df.iloc[indices_test_id_for_metadata].copy()
#                 # Tạo tên file metadata riêng cho audio test in-domain
#                 audio_test_id_metadata_path = os.path.join(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, 'audio_youtube_test_id_metadata.csv')
#                 test_id_metadata_audio_df.to_csv(audio_test_id_metadata_path, index=False)
#                 print(f"  Đã lưu metadata của Test In-Domain cho Audio vào: {audio_test_id_metadata_path}")

#         except ValueError as e_split_test_id_audio:
#             print(f"CẢNH BÁO: Lỗi khi tách Test In-Domain cho Audio: {e_split_test_id_audio}. Sử dụng tất cả cho train/val.")
#             X_mfcc_train_val_list = X_mfcc_all_list
#             y_train_val_encoded = y_encoded_all
#     else:
#         print("CẢNH BÁO: Dữ liệu quá ít hoặc tỷ lệ test bằng 0. Không tách Test In-Domain cho Audio. Sử dụng tất cả cho train/val.")
#         X_mfcc_train_val_list = X_mfcc_all_list
#         y_train_val_encoded = y_encoded_all

#     # Chia phần (Train+Val) của Audio thành Train và Validation
#     print("\n--- Chia (Train+Val) của Audio YouTube thành Train và Validation riêng cho Audio ---")
#     # ... (Logic chia train/val tương tự như fine_tune_fusion_pipeline,
#     #      sử dụng X_mfcc_train_val_list và y_train_val_encoded,
#     #      và cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT)
#     can_split_val_audio = len(y_train_val_encoded) >= 2 and \
#                           (len(y_train_val_encoded) * cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT >= 1 or cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT == 0) and \
#                           (len(y_train_val_encoded) * (1-cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT) >=1 or (1-cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT) == 0) and \
#                           (len(np.unique(y_train_val_encoded)) >=2 or cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT == 0 or (1-cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT)==0)
    
#     if can_split_val_audio and cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT > 0:
#         try:
#             X_train_mfcc, X_val_mfcc, y_train, y_val = train_test_split(
#                 X_mfcc_train_val_list, y_train_val_encoded,
#                 test_size=cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT,
#                 random_state=42, stratify=y_train_val_encoded
#             )
#         except ValueError as e_split_val_audio:
#             print(f"CẢNH BÁO: Lỗi khi chia Validation cho Audio YT: {e_split_val_audio}. Dùng tất cả train_val cho training.")
#             X_train_mfcc = X_mfcc_train_val_list; y_train = y_train_val_encoded
#             X_val_mfcc, y_val = [], np.array([])
#     else:
#         print("CẢNH BÁO: Không đủ dữ liệu hoặc tỷ lệ val bằng 0 cho Audio YT. Dùng tất cả train_val cho training.")
#         X_train_mfcc = X_mfcc_train_val_list; y_train = y_train_val_encoded
#         X_val_mfcc, y_val = [], np.array([])

#     print(f"Số mẫu train audio YouTube cuối cùng: {len(y_train)}")
#     print(f"Số mẫu val audio YouTube cuối cùng: {len(y_val)}")


#     # 4. Đệm Đặc trưng (Padding) - Tính max_len trên tập train MỚI này
#     print("\n--- Đệm MFCC Features cho Audio YouTube (Train/Val) ---")
#     if not X_train_mfcc: print("LỖI: Không có dữ liệu training MFCC để đệm. Dừng."); return
#     X_train_padded, calculated_max_len_yt = pad_features(X_train_mfcc)
#     if X_train_padded.size == 0: print("LỖI: Train features rỗng sau padding."); return
#     print(f"Max sequence length cho Audio YouTube Fine-tune: {calculated_max_len_yt}")
#     print(f"Đang lưu MaxLen Audio YouTube vào: {cfg.AUDIO_YOUTUBE_MAX_LEN_PATH}")
#     os.makedirs(os.path.dirname(cfg.AUDIO_YOUTUBE_MAX_LEN_PATH), exist_ok=True)
#     with open(cfg.AUDIO_YOUTUBE_MAX_LEN_PATH, 'wb') as f: pickle.dump(calculated_max_len_yt, f)

#     if X_val_mfcc: X_val_padded, _ = pad_features(X_val_mfcc, max_len=calculated_max_len_yt)
#     else: X_val_padded = np.array([])

#     # Bây giờ pad cho tập Test In-Domain Audio (nếu có)
#     if X_mfcc_test_id_list:
#         X_mfcc_test_id_padded, _ = pad_features(X_mfcc_test_id_list, max_len=calculated_max_len_yt)
#         # Lưu features Test In-Domain đã đệm (chưa scale)
#         # Scaler sẽ được áp dụng trong script đánh giá
#         with open(os.path.join(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, 'audio_yt_test_id_features_padded.pkl'), 'wb') as f:
#             pickle.dump(X_mfcc_test_id_padded, f)
#         print(f"  Đã lưu PADDED features của Test In-Domain cho Audio (chưa scale).")


#     # 5. Chuẩn hóa Đặc trưng (Scaling) - Fit scaler MỚI trên X_train_padded này
#     print("\n--- Chuẩn hóa MFCC Features cho Audio YouTube (Train/Val) ---")
#     X_train_scaled, audio_yt_scaler = scale_features(X_train_padded)
#     print(f"Đang lưu Audio YouTube Scaler vào: {cfg.AUDIO_YOUTUBE_SCALER_PATH}")
#     os.makedirs(os.path.dirname(cfg.AUDIO_YOUTUBE_SCALER_PATH), exist_ok=True)
#     with open(cfg.AUDIO_YOUTUBE_SCALER_PATH, 'wb') as f: pickle.dump(audio_yt_scaler, f)

#     if X_val_padded.size > 0: X_val_scaled, _ = scale_features(X_val_padded, scaler=audio_yt_scaler)
#     else: X_val_scaled = np.array([])
    
#     validation_data_for_fit = (X_val_scaled, y_val) if X_val_scaled.size > 0 and y_val.size > 0 else None

#     # 6. Xây dựng/Tải và Điều chỉnh Mô hình Audio
#     # ... (Giữ nguyên logic xây dựng và fine-tune 2 giai đoạn từ Câu hỏi 38) ...
#     # ... (Đảm bảo sử dụng cfg.AUDIO_MODEL_SAVE_PATH để tải model gốc) ...
#     # ... (Và cfg.AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE cho số lớp output mới) ...
#     # ... (Lưu model fine-tuned vào cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH) ...
#     print("\n--- Xây dựng/Điều chỉnh Mô hình Audio cho Fine-tuning trên YouTube Data ---")
#     # ... (Toàn bộ code build và train 2 giai đoạn cho audio model như đã cung cấp ở Câu hỏi 38) ...
#     # ... (Bao gồm cả phần đánh giá trên tập validation của audio YouTube) ...
#     # Dưới đây là phần tóm tắt lại logic đó:
#     try:
#         audio_base_model = load_model(cfg.AUDIO_MODEL_SAVE_PATH)
#         if not audio_base_model.built:
#             audio_yt_input_shape_for_build = (calculated_max_len_yt, cfg.AUDIO_N_MFCC)
#             audio_base_model.build(input_shape=(None, *audio_yt_input_shape_for_build))
#     except Exception as e_load: print(f"LỖI tải model audio gốc: {e_load}"); return
#     try:
#         last_internal_layer_name_original_audio = 'dropout_3' # Tên lớp đã xác nhận
#         base_output = audio_base_model.get_layer(last_internal_layer_name_original_audio).output
#     except ValueError: print(f"Lỗi: Không tìm thấy lớp '{last_internal_layer_name_original_audio}'."); audio_base_model.summary(); return
#     new_output_layer = Dense(len(cfg.AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE), activation='softmax', name='new_audio_youtube_output')(base_output)
#     audio_model_for_finetune = Model(inputs=audio_base_model.inputs, outputs=new_output_layer, name='audio_model_youtube_finetune')

#     # Giai đoạn 1
#     print("\n--- Giai đoạn 1 Fine-tune Audio YouTube: Huấn luyện lớp Output mới ---")
#     for layer in audio_model_for_finetune.layers:
#         if layer.name != 'new_audio_youtube_output': layer.trainable = False
#         else: layer.trainable = True
#     audio_model_for_finetune.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.AUDIO_YOUTUBE_FINETUNE_LR_STAGE1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     audio_model_for_finetune.summary(line_length=120)
#     callbacks_ft_s1 = [EarlyStopping(monitor='val_loss' if validation_data_for_fit else 'loss', patience=10, restore_best_weights=True, verbose=1), ModelCheckpoint(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), ReduceLROnPlateau(monitor='val_loss' if validation_data_for_fit else 'loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)]
#     history_ft_s1 = audio_model_for_finetune.fit(X_train_scaled, y_train, validation_data=validation_data_for_fit, epochs=cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE1, batch_size=cfg.AUDIO_YOUTUBE_FINETUNE_BATCH_SIZE, callbacks=callbacks_ft_s1, verbose=1)
#     plot_training_history(history_ft_s1, filename=cfg.AUDIO_YOUTUBE_TRAINING_HISTORY_PLOT_PATH, title_prefix="Audio YT FT Stage 1 - ")

#     # Giai đoạn 2
#     history_ft_s2 = None
#     if cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE2 > 0:
#         print("\n--- Giai đoạn 2 Fine-tune Audio YouTube: Mở băng và huấn luyện sâu hơn ---")
#         try: audio_model_for_finetune_s2 = load_model(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
#         except Exception as e: print(f"Lỗi tải model stage 1: {e}. Dùng model hiện tại."); audio_model_for_finetune_s2 = audio_model_for_finetune
#         for layer in audio_model_for_finetune_s2.layers: layer.trainable = True
#         audio_model_for_finetune_s2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.AUDIO_YOUTUBE_FINETUNE_LR_STAGE2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         audio_model_for_finetune_s2.summary(line_length=120)
#         callbacks_ft_s2 = [EarlyStopping(monitor='val_loss' if validation_data_fit else 'loss', patience=8, restore_best_weights=True, verbose=1), ModelCheckpoint(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), ReduceLROnPlateau(monitor='val_loss' if validation_data_for_fit else 'loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)]
#         start_epoch_s2 = 0
#         if history_ft_s1 and hasattr(history_ft_s1, 'epoch') and history_ft_s1.epoch: start_epoch_s2 = history_ft_s1.epoch[-1] + 1
#         final_epochs_s2_target = start_epoch_s2 + cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE2
#         history_ft_s2 = audio_model_for_finetune_s2.fit(X_train_scaled, y_train, validation_data=validation_data_for_fit, epochs=final_epochs_s2_target, initial_epoch=start_epoch_s2, batch_size=cfg.AUDIO_YOUTUBE_FINETUNE_BATCH_SIZE, callbacks=callbacks_ft_s2, verbose=1)
#         if history_ft_s2 and history_ft_s2.history: plot_training_history(history_ft_s2, filename=os.path.join(cfg.AUDIO_YOUTUBE_REPORTS_FIGURES_DIR, 'audio_yt_ft_history_stage2.png'), title_prefix="Audio YT FT Stage 2 - ", initial_epoch_offset=start_epoch_s2)
#         else: print("Cảnh báo: Không có history Giai đoạn 2 để vẽ.")
#     else: print("\nBỏ qua Giai đoạn 2 Fine-tune Audio YouTube.")

#     # Đánh giá trên Validation Set của Audio YouTube
#     if validation_data_for_fit:
#         print("\n--- Đánh giá Model Audio Fine-tuned trên Tập Validation (của bộ Audio YouTube) ---")
#         try: final_audio_yt_model = load_model(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
#         except Exception as e: print(f"Lỗi tải model audio fine-tuned cuối cùng: {e}"); return
#         val_pred_probs_audio_yt = final_audio_yt_model.predict(X_val_scaled)
#         val_pred_classes_audio_yt = np.argmax(val_pred_probs_audio_yt, axis=1)
#         print("\nClassification Report (Audio YouTube Fine-tune Validation Set):")
#         print(classification_report(y_val, val_pred_classes_audio_yt, target_names=list(audio_yt_label_encoder.classes_), labels=range(len(audio_yt_label_encoder.classes_)), zero_division=0))
#         plot_custom_confusion_matrix(y_val, val_pred_classes_audio_yt, list(audio_yt_label_encoder.classes_), filename=cfg.AUDIO_YOUTUBE_CONFUSION_MATRIX_VAL_PLOT_PATH, title_prefix="Audio YT FT Validation - ")
#     else: print("\nKhông có dữ liệu validation để đánh giá mô hình audio fine-tuned.")

#     print("===== AUDIO MODEL FINE-TUNING PIPELINE ON YOUTUBE DATA FINISHED =====")
#     if X_mfcc_test_id_list:
#         print(f"Tập Test In-Domain cho Audio (features padded, labels, và metadata) đã được lưu.")
#         print(f"  Audio Features (Padded): {os.path.join(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, 'audio_yt_test_id_features_padded.pkl')}")
#         print(f"  Audio Labels: {os.path.join(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, 'audio_yt_test_id_labels.pkl')}")
#         print(f"  Audio Metadata: {os.path.join(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, 'audio_yt_test_id_metadata.csv')}")
#     else:
#         print("Không có tập Test In-Domain nào được tạo ra cho Audio.")


# if __name__ == '__main__':
#     required_for_audio_yt_ft = [
#         cfg.AUDIO_MODEL_SAVE_PATH,
#         cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE
#     ]
#     missing_reqs = [f for f in required_for_audio_yt_ft if not os.path.exists(f)]
#     if missing_reqs:
#         print("LỖI: Thiếu các file cần thiết để chạy pipeline fine-tune audio YouTube:")
#         for f_path in missing_reqs: print(f" - {f_path}")
#     else:
#         run_audio_youtube_finetuning_pipeline()

# Drone_Detection_Project/src/training/fine_tune_audio_youtube_pipeline.py
import os
import glob
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout # Đảm bảo Input và Dropout được import
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd

# Import từ các module trong src
try:
    from src.config_loader.loader import get_config
    from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from src.evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix
except ImportError:
    import sys
    current_dir_ft_audio_yt = os.path.dirname(os.path.abspath(__file__))
    src_dir_ft_audio_yt = os.path.dirname(current_dir_ft_audio_yt)
    project_root_ft_audio_yt = os.path.dirname(src_dir_ft_audio_yt)
    if project_root_ft_audio_yt not in sys.path: sys.path.insert(0, project_root_ft_audio_yt)
    if src_dir_ft_audio_yt not in sys.path: sys.path.insert(0, src_dir_ft_audio_yt)

    from config_loader.loader import get_config
    from data_processing.audio_utils import extract_mfcc, pad_features, scale_features
    from evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix

cfg = get_config()

def load_audio_youtube_segments_for_finetune(metadata_file_path, segments_base_dir):
    """
    Tải dữ liệu audio segment .wav và nhãn từ file metadata.
    Trả về list các đường dẫn file audio, list nhãn text, và DataFrame metadata gốc cùng list các ID/index gốc.
    """
    print(f"--- Đang tải danh sách audio segments cho fine-tune từ metadata: {metadata_file_path} ---")
    if not os.path.exists(metadata_file_path):
        print(f"LỖI: File metadata audio YouTube không tồn tại: {metadata_file_path}")
        return [], [], None, []
    try:
        metadata_df = pd.read_csv(metadata_file_path)
        print(f"Đã đọc {len(metadata_df)} dòng từ metadata.")
    except Exception as e:
        print(f"Lỗi khi đọc file metadata audio YouTube: {e}")
        return [], [], None, []

    all_audio_filepaths = []
    all_labels_text = []
    all_original_indices_or_ids = []

    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Scanning audio segments for YT fine-tune"):
        try:
            audio_segment_rel_path = str(row['audio_segment_file']).strip()
            label = str(row['label']).strip()
            audio_segment_abs_path = os.path.join(segments_base_dir, audio_segment_rel_path)

            if not os.path.exists(audio_segment_abs_path):
                continue
            
            if os.path.getsize(audio_segment_abs_path) < 100:
                continue

            all_audio_filepaths.append(audio_segment_abs_path)
            all_labels_text.append(label)
            all_original_indices_or_ids.append(index)
        except Exception as e:
            print(f"Lỗi khi quét segment {row.get('audio_segment_file', f'index {index}')}: {e}")
            continue
    
    if not all_labels_text:
        print("Không tìm thấy audio segment nào hợp lệ từ metadata.")
        return [], [], None, []

    print(f"Đã quét {len(all_labels_text)} audio segments hợp lệ từ dữ liệu YouTube.")
    return all_audio_filepaths, all_labels_text, metadata_df, all_original_indices_or_ids


def process_audio_list_to_mfcc_features(audio_filepaths_list, sample_rate, n_mfcc, n_fft, hop_length,
                                        max_len_for_padding=None, scaler_to_use=None,
                                        fit_scaler_and_max_len=False):
    """
    Xử lý một danh sách các đường dẫn file audio thành features MFCC đã pad và scale.
    """
    all_mfcc_temp = []
    valid_indices_after_extraction = []

    print(f"  Bắt đầu trích xuất MFCC cho {len(audio_filepaths_list)} files...")
    for idx, audio_path in enumerate(tqdm(audio_filepaths_list, desc="Extracting MFCC")):
        try:
            audio_data, sr_loaded = librosa.load(audio_path, sr=sample_rate)
            if len(audio_data) < int(cfg.AUDIO_SEGMENT_DURATION * sample_rate * 0.1):
                continue
            mfccs = extract_mfcc(audio_data, sample_rate, n_mfcc, n_fft, hop_length)
            all_mfcc_temp.append(mfccs)
            valid_indices_after_extraction.append(idx)
        except Exception as e:
            print(f"    Lỗi khi trích xuất MFCC từ {os.path.basename(audio_path)}: {e}")
            continue
    
    if not all_mfcc_temp:
        print("    Không có MFCC nào được trích xuất.")
        return np.array([]), None, None, []

    if fit_scaler_and_max_len or max_len_for_padding is None:
        if not all_mfcc_temp: # Kiểm tra lại sau khi vòng lặp trích xuất
            print("Lỗi: Không có MFCC nào để tính max_len (trong fit_scaler_and_max_len).")
            return np.array([]), None, None, []
        calculated_max_len = max(feat.shape[0] for feat in all_mfcc_temp if feat.ndim > 0 and feat.shape[0] > 0)
        print(f"    Max_len tính từ dữ liệu hiện tại: {calculated_max_len}")
    else:
        calculated_max_len = max_len_for_padding
        print(f"    Sử dụng max_len cung cấp: {calculated_max_len}")

    padded_mfcc_list = []
    final_valid_indices = []

    for i, feat in enumerate(all_mfcc_temp):
        original_file_idx = valid_indices_after_extraction[i]
        padded_feat, _ = pad_features([feat], max_len=calculated_max_len)
        if padded_feat.size > 0:
            padded_mfcc_list.append(padded_feat[0])
            final_valid_indices.append(original_file_idx)

    if not padded_mfcc_list:
        print("    Không có MFCC nào sau padding.")
        return np.array([]), None, None, []

    X_padded = np.array(padded_mfcc_list)

    if fit_scaler_and_max_len or scaler_to_use is None:
        print("    Fit scaler mới từ dữ liệu padding hiện tại.")
        X_scaled, fitted_scaler = scale_features(X_padded) # scale_features sẽ tạo scaler mới nếu scaler_to_use=None
    else:
        print("    Sử dụng scaler đã cung cấp.")
        X_scaled, fitted_scaler = scale_features(X_padded, scaler=scaler_to_use) # fitted_scaler sẽ là scaler_to_use

    if X_scaled.size == 0: print("    Dữ liệu rỗng sau scaling."); return np.array([]), None, None, []
    
    return X_scaled, fitted_scaler, calculated_max_len, final_valid_indices


def run_audio_youtube_finetuning_pipeline():
    print("===== STARTING AUDIO MODEL FINE-TUNING PIPELINE ON YOUTUBE DATA (WITH TEST IN-DOMAIN SPLIT FOR AUDIO) =====")
    if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories):
         cfg.ensure_output_directories()

    # 1. Tải Danh sách File và Nhãn
    all_audio_filepaths, y_text_all, full_original_metadata_df, original_indices_all = \
        load_audio_youtube_segments_for_finetune(
            metadata_file_path=cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE,
            segments_base_dir=cfg.AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR
        )
    if not y_text_all:
        print("Không có dữ liệu audio từ YouTube để fine-tune. Dừng pipeline.")
        return

    # 2. Mã hóa Nhãn
    audio_yt_label_encoder = LabelEncoder()
    audio_yt_label_encoder.fit(cfg.AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE)
    print(f"Audio YouTube Fine-tune LabelEncoder classes: {list(audio_yt_label_encoder.classes_)}")
    try:
        y_encoded_all = audio_yt_label_encoder.transform(y_text_all)
    except ValueError as e:
        print(f"LỖI Mã hóa Nhãn cho Audio YouTube: {e}...")
        return
    print(f"Đang lưu Audio YouTube LabelEncoder vào: {cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH}")
    os.makedirs(os.path.dirname(cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH), exist_ok=True)
    with open(cfg.AUDIO_YOUTUBE_LABEL_ENCODER_PATH, 'wb') as f: pickle.dump(audio_yt_label_encoder, f)

    # 3. Chia Dữ liệu: Test (30%), phần còn lại Train (80%) / Val (20%)
    print("\n--- Chia dữ liệu Audio YouTube thành (Train+Val) và Test In-Domain ---")
    all_audio_filepaths_np = np.array(all_audio_filepaths)
    original_indices_np = np.array(original_indices_all)
    
    TEST_SPLIT_RATIO_AUDIO_YT = 0.30 # Định nghĩa tỷ lệ test
    VALIDATION_SPLIT_RATIO_AUDIO_YT = cfg.AUDIO_YOUTUBE_VALIDATION_SPLIT # Lấy từ config, ví dụ 0.20

    paths_train_val, paths_test_id, \
    y_train_val_encoded, y_test_id_encoded, \
    indices_train_val_metadata, indices_test_id_metadata = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))


    can_split_test_audio = len(y_encoded_all) >= 2 and \
                           (len(y_encoded_all) * TEST_SPLIT_RATIO_AUDIO_YT >= 1 or TEST_SPLIT_RATIO_AUDIO_YT == 0) and \
                           (len(y_encoded_all) * (1-TEST_SPLIT_RATIO_AUDIO_YT) >= 1 or (1-TEST_SPLIT_RATIO_AUDIO_YT) == 0) and \
                           (len(np.unique(y_encoded_all)) >= 2 or TEST_SPLIT_RATIO_AUDIO_YT == 0 or (1-TEST_SPLIT_RATIO_AUDIO_YT) == 0)

    if can_split_test_audio and TEST_SPLIT_RATIO_AUDIO_YT > 0:
        try:
            paths_train_val, paths_test_id, \
            y_train_val_encoded, y_test_id_encoded, \
            indices_train_val_metadata, indices_test_id_metadata = train_test_split(
                all_audio_filepaths_np, y_encoded_all, original_indices_np,
                test_size=TEST_SPLIT_RATIO_AUDIO_YT,
                random_state=42,
                stratify=y_encoded_all
            )
            print(f"Số mẫu Test In-Domain cho Audio: {len(y_test_id_encoded)}")
        except ValueError as e_split_test_id_audio:
            print(f"CẢNH BÁO: Lỗi khi tách Test Audio YT: {e_split_test_id_audio}. Sử dụng tất cả cho train/val.")
            paths_train_val, y_train_val_encoded, indices_train_val_metadata = all_audio_filepaths_np, y_encoded_all, original_indices_np
    else:
        print("CẢNH BÁO: Không tách Test In-Domain cho Audio. Sử dụng tất cả cho train/val.")
        paths_train_val, y_train_val_encoded, indices_train_val_metadata = all_audio_filepaths_np, y_encoded_all, original_indices_np

    # Xử lý Đặc trưng cho Tập Train+Validation (để fit scaler và tính max_len)
    print("\n--- Xử lý (MFCC, Pad, Scale) cho tập Train+Validation của Audio YouTube ---")
    X_train_val_scaled, scaler_yt, max_len_yt, valid_indices_tv = process_audio_list_to_mfcc_features(
        list(paths_train_val), cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC,
        cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH,
        fit_scaler_and_max_len=True
    )
    if X_train_val_scaled.size == 0: print("Lỗi: Không có features sau khi xử lý tập train_val. Dừng."); return
    y_train_val_encoded = y_train_val_encoded[valid_indices_tv]
    # Cập nhật indices_train_val_metadata nếu cần (ít quan trọng ở bước này)

    print(f"Lưu Scaler đã fit trên Audio YouTube Train/Val vào: {cfg.AUDIO_YOUTUBE_SCALER_PATH}")
    with open(cfg.AUDIO_YOUTUBE_SCALER_PATH, 'wb') as f: pickle.dump(scaler_yt, f)
    print(f"Lưu MaxLen tính từ Audio YouTube Train/Val vào: {cfg.AUDIO_YOUTUBE_MAX_LEN_PATH}")
    with open(cfg.AUDIO_YOUTUBE_MAX_LEN_PATH, 'wb') as f: pickle.dump(max_len_yt, f)

    # Xử lý Đặc trưng cho Tập Test In-Domain (nếu có)
    X_test_id_scaled = np.array([])
    y_test_id_encoded_final = np.array([]) # Nhãn cuối cùng cho test sau khi xử lý feature
    indices_test_id_metadata_final = np.array([])

    if paths_test_id.size > 0:
        print("\n--- Xử lý (MFCC, Pad, Scale) cho tập Test In-Domain của Audio YouTube ---")
        X_test_id_scaled_temp, _, _, valid_indices_test = process_audio_list_to_mfcc_features(
            list(paths_test_id), cfg.AUDIO_SAMPLE_RATE, cfg.AUDIO_N_MFCC,
            cfg.AUDIO_N_FFT, cfg.AUDIO_HOP_LENGTH,
            max_len_for_padding=max_len_yt, scaler_to_use=scaler_yt
        )
        if X_test_id_scaled_temp.size > 0:
            X_test_id_scaled = X_test_id_scaled_temp
            y_test_id_encoded_final = y_test_id_encoded[valid_indices_test]
            indices_test_id_metadata_final = indices_test_id_metadata[valid_indices_test]

            test_data_save_dir = os.path.join(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'test_in_domain_audio_data')
            os.makedirs(test_data_save_dir, exist_ok=True)
            with open(os.path.join(test_data_save_dir, 'X_test_audio_yt_mfcc_scaled.pkl'), 'wb') as f: pickle.dump(X_test_id_scaled, f)
            with open(os.path.join(test_data_save_dir, 'y_test_audio_yt_encoded.pkl'), 'wb') as f: pickle.dump(y_test_id_encoded_final, f)
            
            if full_original_metadata_df is not None and indices_test_id_metadata_final.size > 0:
                test_id_metadata_audio_df = full_original_metadata_df.iloc[indices_test_id_metadata_final].copy()
                original_paths_for_test_metadata = paths_test_id[valid_indices_test]
                test_id_metadata_audio_df['original_segment_filepath_relative'] = [
                    os.path.relpath(p, start=cfg.AUDIO_YOUTUBE_FINETUNE_SEGMENTS_DIR) for p in original_paths_for_test_metadata
                ]
                audio_test_id_metadata_path = os.path.join(test_data_save_dir, 'test_audio_yt_metadata.csv')
                test_id_metadata_audio_df.to_csv(audio_test_id_metadata_path, index=False)
            print(f"    Đã lưu dữ liệu Test Audio YT (MFCC SCALED, labels, metadata) vào: {test_data_save_dir}")


    # Chia Train và Validation từ X_train_val_scaled
    print("\n--- Chia (Train+Val) đã xử lý của Audio YouTube thành Train và Validation ---")
    validation_audio_data_for_fit = None
    X_train_audio_final, y_train_final = X_train_val_scaled, y_train_val_encoded # Mặc định nếu không chia val

    can_split_val_audio_final = len(y_train_val_encoded) >= 2 and \
                                (len(y_train_val_encoded) * VALIDATION_SPLIT_RATIO_AUDIO_YT >= 1 or VALIDATION_SPLIT_RATIO_AUDIO_YT == 0) and \
                                (len(y_train_val_encoded) * (1-VALIDATION_SPLIT_RATIO_AUDIO_YT) >=1 or (1-VALIDATION_SPLIT_RATIO_AUDIO_YT) == 0) and \
                                (len(np.unique(y_train_val_encoded)) >=2 or VALIDATION_SPLIT_RATIO_AUDIO_YT == 0 or (1-VALIDATION_SPLIT_RATIO_AUDIO_YT)==0)

    if can_split_val_audio_final and VALIDATION_SPLIT_RATIO_AUDIO_YT > 0:
        try:
            X_train_audio_final, X_val_audio_final, \
            y_train_final, y_val_final = train_test_split(
                X_train_val_scaled, y_train_val_encoded,
                test_size=VALIDATION_SPLIT_RATIO_AUDIO_YT,
                random_state=42, stratify=y_train_val_encoded
            )
            if X_val_audio_final.size > 0 and y_val_final.size > 0:
                validation_audio_data_for_fit = (X_val_audio_final, y_val_final)
            else:
                print("  Cảnh báo: Tập validation audio rỗng sau khi chia cuối cùng.")
        except ValueError as e_split_val_final:
            print(f"CẢNH BÁO: Lỗi khi chia Validation cuối cùng cho Audio YT: {e_split_val_final}.")
    else:
        print("CẢNH BÁO: Không đủ dữ liệu hoặc tỷ lệ val bằng 0. Toàn bộ train_val được dùng cho training.")

    print(f"Số mẫu Train Audio YT cuối cùng: {len(y_train_final)}")
    if validation_audio_data_for_fit: print(f"Số mẫu Validation Audio YT cuối cùng: {len(y_val_final)}")
    else: print("Không có tập Validation Audio YT cuối cùng.")
    if X_train_audio_final.size > 0 : print(f"Shape X_train_audio_final: {X_train_audio_final.shape}")
    else: print("LỖI: Tập training cuối cùng rỗng!"); return

    # 4. Xây dựng và Fine-tune Mô hình Audio
    print("\n--- Xây dựng/Điều chỉnh Mô hình Audio cho Fine-tuning trên YouTube Data ---")
    try:
        audio_base_model = load_model(cfg.AUDIO_MODEL_SAVE_PATH)
        if not audio_base_model.built and X_train_audio_final.ndim == 3:
            current_input_shape_for_build = (X_train_audio_final.shape[1], X_train_audio_final.shape[2])
            audio_base_model.build(input_shape=(None, *current_input_shape_for_build))
            print(f"Đã build audio_base_model với input_shape: {(None, *current_input_shape_for_build)}")
    except Exception as e_load: print(f"LỖI tải model audio gốc: {e_load}"); return
    
    try:
        last_internal_layer_name_original_audio = cfg.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION # Đã xác nhận từ summary
        base_output = audio_base_model.get_layer(last_internal_layer_name_original_audio).output
    except ValueError:
        print(f"Lỗi: Không tìm thấy lớp '{last_internal_layer_name_original_audio}'. In summary:");
        try: audio_base_model.summary(line_length=120)
        except: print("Không thể in summary của base_audio_model.")
        return

    num_youtube_classes = len(cfg.AUDIO_CLASSES_FOR_YOUTUBE_FINETUNE)
    new_output_layer = Dense(num_youtube_classes, activation='softmax', name='new_audio_youtube_output')(base_output)
    audio_model_for_finetune = Model(inputs=audio_base_model.inputs, outputs=new_output_layer, name='audio_model_youtube_finetune')

    # Giai đoạn 1
    print("\n--- Giai đoạn 1 Fine-tune Audio YouTube: Huấn luyện lớp Output mới ---")
    for layer in audio_model_for_finetune.layers:
        if layer.name != 'new_audio_youtube_output' and layer.name in [l.name for l in audio_base_model.layers]:
            layer.trainable = False
        else:
            layer.trainable = True
    audio_model_for_finetune.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.AUDIO_YOUTUBE_FINETUNE_LR_STAGE1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    audio_model_for_finetune.summary(line_length=120)
    callbacks_ft_s1 = [EarlyStopping(monitor='val_loss' if validation_audio_data_for_fit else 'loss', patience=10, restore_best_weights=True, verbose=1), ModelCheckpoint(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_audio_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), ReduceLROnPlateau(monitor='val_loss' if validation_audio_data_for_fit else 'loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)]
    history_ft_s1 = audio_model_for_finetune.fit(X_train_audio_final, y_train_final, validation_data=validation_audio_data_for_fit, epochs=cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE1, batch_size=cfg.AUDIO_YOUTUBE_FINETUNE_BATCH_SIZE, callbacks=callbacks_ft_s1, verbose=1)
    # Lưu lịch sử stage 1 riêng nếu muốn
    plot_training_history(history_ft_s1, filename=os.path.join(cfg.AUDIO_YOUTUBE_REPORTS_FIGURES_DIR, 'audio_yt_ft_history_stage1.png'), title_prefix="Audio YT FT Stage 1 - ")

    # Giai đoạn 2
    history_ft_s2 = None
    if cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE2 > 0:
        print("\n--- Giai đoạn 2 Fine-tune Audio YouTube: Mở băng và huấn luyện sâu hơn ---")
        try: audio_model_for_finetune_s2 = load_model(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
        except Exception as e: print(f"Lỗi tải model stage 1: {e}. Dùng model hiện tại."); audio_model_for_finetune_s2 = audio_model_for_finetune
        
        print("Mở băng các lớp trong nhánh audio gốc (trừ lớp input):")
        opened_layers_count = 0
        for layer in audio_model_for_finetune_s2.layers:
            if layer.name in [l.name for l in audio_base_model.layers] and not isinstance(layer, tf.keras.layers.InputLayer):
                layer.trainable = True
                opened_layers_count +=1
        print(f"  Đã mở băng {opened_layers_count} lớp từ mô hình audio gốc.")
        
        audio_model_for_finetune_s2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.AUDIO_YOUTUBE_FINETUNE_LR_STAGE2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        audio_model_for_finetune_s2.summary(line_length=120)
        callbacks_ft_s2 = [EarlyStopping(monitor='val_loss' if validation_audio_data_for_fit else 'loss', patience=15, restore_best_weights=True, verbose=1), ModelCheckpoint(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_audio_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), ReduceLROnPlateau(monitor='val_loss' if validation_audio_data_for_fit else 'loss', factor=0.3, patience=7, min_lr=1e-7, verbose=1)]
        start_epoch_s2 = 0
        if history_ft_s1 and hasattr(history_ft_s1, 'epoch') and history_ft_s1.epoch : start_epoch_s2 = history_ft_s1.epoch[-1] + 1
        
        # Tính số epoch thực sự cần chạy cho stage 2
        epochs_to_run_s2 = cfg.AUDIO_YOUTUBE_FINETUNE_EPOCHS_STAGE2
        
        print(f"  Bắt đầu huấn luyện Giai đoạn 2 từ epoch tổng thể {start_epoch_s2} với {epochs_to_run_s2} epochs mục tiêu cho giai đoạn này...")
        history_ft_s2 = audio_model_for_finetune_s2.fit(
            X_train_audio_final, y_train_final,
            validation_data=validation_audio_data_for_fit,
            epochs=start_epoch_s2 + epochs_to_run_s2, # Tổng số epoch từ đầu
            initial_epoch=start_epoch_s2,
            batch_size=cfg.AUDIO_YOUTUBE_FINETUNE_BATCH_SIZE,
            callbacks=callbacks_ft_s2,
            verbose=1
        )
        if history_ft_s2 and history_ft_s2.history:
            plot_training_history(history_ft_s2, filename=os.path.join(cfg.AUDIO_YOUTUBE_REPORTS_FIGURES_DIR, 'audio_yt_ft_history_stage2.png'), title_prefix="Audio YT FT Stage 2 - ", initial_epoch_offset=start_epoch_s2)
    else: print("\nBỏ qua Giai đoạn 2 Fine-tune Audio YouTube.")


    # 5. Đánh giá cuối cùng trên Tập Validation
    final_history_audio_yt = history_ft_s1 # Mặc định là history của stage 1
    if history_ft_s2 and history_ft_s2.history: # Nếu stage 2 chạy và có history
        # Gộp history nếu muốn vẽ biểu đồ tổng thể
        # Đây là một cách đơn giản, bạn có thể cần xử lý phức tạp hơn nếu muốn gộp chính xác
        # For plotting combined history:
        combined_history = {}
        if history_ft_s1 and history_ft_s1.history:
            for key, value in history_ft_s1.history.items():
                combined_history[key] = list(value) # Chuyển sang list
        if history_ft_s2 and history_ft_s2.history: # Chỉ gộp nếu history_ft_s2 tồn tại
            for key, value in history_ft_s2.history.items():
                if key in combined_history:
                    combined_history[key].extend(value)
                else: # Trường hợp stage 1 không chạy (epochs=0)
                    combined_history[key] = list(value)
        
        # Tạo một đối tượng history giả để plot
        class MockHistory: pass
        final_history_plot_obj = MockHistory()
        final_history_plot_obj.history = combined_history
        final_history_plot_obj.epoch = list(range(len(combined_history.get('loss',[]))))


        plot_training_history(final_history_plot_obj, filename=cfg.AUDIO_YOUTUBE_TRAINING_HISTORY_PLOT_PATH,
                              title_prefix="Audio YT FT Combined - ")
        print(f"Lịch sử huấn luyện Audio YouTube Fine-tuned (Combined) đã lưu: {cfg.AUDIO_YOUTUBE_TRAINING_HISTORY_PLOT_PATH}")
    elif final_history_audio_yt and final_history_audio_yt.history : # Chỉ có stage 1
         plot_training_history(final_history_audio_yt, filename=cfg.AUDIO_YOUTUBE_TRAINING_HISTORY_PLOT_PATH,
                              title_prefix="Audio YT FT Stage 1 - ")
         print(f"Lịch sử huấn luyện Audio YouTube Fine-tuned (Stage 1) đã lưu: {cfg.AUDIO_YOUTUBE_TRAINING_HISTORY_PLOT_PATH}")


    if validation_audio_data_for_fit and y_val_final.size > 0:
        print("\n--- Đánh giá Model Audio Fine-tuned trên Tập Validation (của bộ Audio YouTube) ---")
        try: best_audio_yt_model = load_model(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_SAVE_PATH)
        except Exception as e: print(f"Lỗi tải model audio fine-tuned cuối cùng: {e}"); return
        
        val_pred_probs_audio_yt = best_audio_yt_model.predict(X_val_audio_final)
        val_pred_classes_audio_yt = np.argmax(val_pred_probs_audio_yt, axis=1)
        print("\nClassification Report (Audio YouTube Fine-tune Validation Set):")
        target_names_audio_report = list(audio_yt_label_encoder.classes_)
        labels_audio_report = list(range(len(target_names_audio_report))) # Phải là list các số nguyên
        
        report_val_str = classification_report(y_val_final, val_pred_classes_audio_yt, target_names=target_names_audio_report, labels=labels_audio_report, zero_division=0)
        print(report_val_str)
        report_audio_yt_val_path = os.path.join(cfg.AUDIO_YOUTUBE_REPORTS_METRICS_DIR, "audio_yt_finetuned_classification_report_val.txt")
        with open(report_audio_yt_val_path, 'w') as f: f.write(report_val_str)
        print(f"Đã lưu Classification Report (Validation) vào: {report_audio_yt_val_path}")
        
        plot_custom_confusion_matrix(y_val_final, val_pred_classes_audio_yt, target_names_audio_report,
                                     filename=cfg.AUDIO_YOUTUBE_CONFUSION_MATRIX_VAL_PLOT_PATH,
                                     title="CM - Audio Fine-tuned on YT (Validation Set)")
    else: print("\nKhông có dữ liệu validation audio để đánh giá mô hình sau fine-tuning.")

    print("===== AUDIO MODEL FINE-TUNING PIPELINE ON YOUTUBE DATA FINISHED =====")
    if X_test_id_scaled.size > 0:
        print(f"\nDữ liệu Test In-Domain cho Audio (features SCALED, labels, và metadata) đã được lưu.")
        print(f"  Audio Features (Scaled): {os.path.join(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'test_in_domain_audio_data', 'X_test_audio_yt_mfcc_scaled.pkl')}")
        print(f"  Audio Labels: {os.path.join(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'test_in_domain_audio_data', 'y_test_audio_yt_encoded.pkl')}")
        print(f"  Audio Metadata: {os.path.join(cfg.AUDIO_YOUTUBE_FINETUNED_MODEL_DIR, 'test_in_domain_audio_data', 'test_audio_yt_metadata.csv')}")
    else:
        print("\nKhông có tập Test In-Domain nào được tạo ra hoặc xử lý thành công cho Audio.")


if __name__ == '__main__':
    required_for_audio_yt_ft = [
        cfg.AUDIO_MODEL_SAVE_PATH, # Mô hình audio gốc
        cfg.AUDIO_YOUTUBE_FINETUNE_METADATA_FILE # Metadata chỉ đến file wav segment từ YouTube
    ]
    missing_reqs = [f for f in required_for_audio_yt_ft if not os.path.exists(f)]
    if missing_reqs:
        print("LỖI: Thiếu các file cần thiết để chạy pipeline fine-tune audio YouTube:")
        for f_path in missing_reqs: print(f" - {f_path}")
    else:
        run_audio_youtube_finetuning_pipeline()