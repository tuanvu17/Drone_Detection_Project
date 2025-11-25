# # Drone_Detection_Project/src/training/fine_tune_fusion_pipeline.py
# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Model
# from tensorflow.keras.layers import Dense # Đảm bảo import Dense
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm
# from sklearn.metrics import classification_report
# import pandas as pd

# try:
#     from src.config_loader.loader import get_config
#     from src.model_architectures.fusion_model import build_early_fusion_model
#     from src.evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix
# except ImportError:
#     import sys
#     current_dir_ft_pipe = os.path.dirname(os.path.abspath(__file__))
#     src_dir_ft_pipe = os.path.dirname(current_dir_ft_pipe)
#     project_root_ft_pipe = os.path.dirname(src_dir_ft_pipe)
#     if project_root_ft_pipe not in sys.path: sys.path.insert(0, project_root_ft_pipe)
#     if src_dir_ft_pipe not in sys.path: sys.path.insert(0, src_dir_ft_pipe)
#     from config_loader.loader import get_config
#     from model_architectures.fusion_model import build_early_fusion_model
#     from evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix

# cfg = get_config()

# def load_fine_tune_data_for_fusion_pipeline(metadata_file_path, paired_data_base_dir):
#     print(f"--- Đang tải dữ liệu fine-tune đã xử lý cho FUSION từ metadata: {metadata_file_path} ---")
#     if not os.path.exists(metadata_file_path):
#         print(f"LỖI: File metadata không tồn tại: {metadata_file_path}")
#         return None, None, None, None
#     try:
#         full_original_metadata_df = pd.read_csv(metadata_file_path)
#     except Exception as e:
#         print(f"Lỗi khi đọc file metadata: {e}")
#         return None, None, None, None

#     all_audio_features, all_vcam_features, all_labels_text = [], [], []
#     # print(f"Tìm thấy {len(full_original_metadata_df)} entries trong metadata.")

#     for index, row in tqdm(full_original_metadata_df.iterrows(), total=full_original_metadata_df.shape[0], desc="Loading paired features for fusion"):
#         try:
#             audio_mfcc_rel_path = row['audio_mfcc_file'] # Đường dẫn đến file .npy MFCC đã xử lý
#             vcam_features_rel_path = row['vcam_features_file'] # Đường dẫn đến file .npy VCam feature đã xử lý
#             label = str(row['label']).strip()

#             audio_mfcc_abs_path = os.path.join(paired_data_base_dir, audio_mfcc_rel_path)
#             vcam_features_abs_path = os.path.join(paired_data_base_dir, vcam_features_rel_path)

#             if not os.path.exists(audio_mfcc_abs_path) or not os.path.exists(vcam_features_abs_path):
#                 # print(f"Cảnh báo: Thiếu file feature cho segment {row.get('segment_id', 'N/A')}. Audio: {audio_mfcc_abs_path}, VCam: {vcam_features_abs_path}")
#                 continue

#             audio_feat = np.load(audio_mfcc_abs_path)
#             vcam_feat = np.load(vcam_features_abs_path)

#             # Kiểm tra shape nếu cần, ví dụ:
#             if audio_feat.shape[1] != cfg.AUDIO_N_MFCC:
#                 # print(f"Cảnh báo: Audio feature shape không đúng {audio_feat.shape} cho {audio_mfcc_abs_path}")
#                 continue
#             if vcam_feat.shape[0] != cfg.FUSION_VCAM_FEATURE_DIM:
#                 # print(f"Cảnh báo: VCam feature shape không đúng {vcam_feat.shape} cho {vcam_features_abs_path}")
#                 continue

#             all_audio_features.append(audio_feat)
#             all_vcam_features.append(vcam_feat)
#             all_labels_text.append(label)
#         except Exception as e:
#             # print(f"Lỗi khi tải features cho segment {row.get('segment_id', 'N/A')}: {e}")
#             continue

#     if not all_labels_text:
#         print("Không tải được mẫu dữ liệu fine-tune nào cho fusion.")
#         return None, None, None, None

#     print(f"Đã tải thành công {len(all_labels_text)} cặp đặc trưng (Audio MFCC, VCam) cho fusion.")
#     return np.array(all_audio_features), np.array(all_vcam_features), all_labels_text, full_original_metadata_df


# def run_fusion_finetuning_pipeline():
#     print("===== STARTING EARLY FUSION MODEL FINE-TUNING PIPELINE (VCam + Audio) =====")
#     if hasattr(cfg, 'ensure_output_directories'): cfg.ensure_output_directories()

#     X_audio_all, X_vcam_all, y_labels_text_all, full_original_metadata_df = load_fine_tune_data_for_fusion_pipeline(
#         metadata_file_path=cfg.FINETUNE_MULTICLASS_METADATA_FILE,
#         paired_data_base_dir=cfg.FINETUNE_PAIRED_DATA_BASE_DIR
#     )

#     if X_audio_all is None or len(y_labels_text_all) == 0:
#         print("Không có dữ liệu fine-tune để huấn luyện. Dừng pipeline.")
#         return

#     print(f"\n--- Mã hóa Nhãn cho Fusion ({len(cfg.FUSION_CLASS_NAMES)} lớp) ---")
#     fusion_label_encoder = LabelEncoder()
#     try:
#         # QUAN TRỌNG: Fit LabelEncoder trên MASTER_CLASS_LIST_FUSION hoặc FUSION_CLASS_NAMES
#         # để đảm bảo tất cả các lớp mục tiêu đều được biết đến và có thứ tự nhất quán.
#         fusion_label_encoder.fit(cfg.FUSION_CLASS_NAMES)
#         y_encoded_all = fusion_label_encoder.transform(y_labels_text_all)
#         print(f"  Các lớp trong Fusion LabelEncoder: {list(fusion_label_encoder.classes_)}")
#     except ValueError as e:
#         print(f"LỖI Mã hóa Nhãn cho Fusion: {e}")
#         print(f"  Các nhãn trong dữ liệu: {np.unique(y_labels_text_all)}")
#         print(f"  Các lớp dự kiến (FUSION_CLASS_NAMES): {cfg.FUSION_CLASS_NAMES}")
#         return

#     print(f"Đang lưu Fusion LabelEncoder vào: {cfg.FUSION_LABEL_ENCODER_PATH}")
#     os.makedirs(os.path.dirname(cfg.FUSION_LABEL_ENCODER_PATH), exist_ok=True)
#     with open(cfg.FUSION_LABEL_ENCODER_PATH, 'wb') as f: pickle.dump(fusion_label_encoder, f)


#     print("\n--- Chia dữ liệu fine-tune thành Train/Validation và Test In-Domain ---")
#     all_indices = np.arange(len(y_encoded_all)) # Lấy index để có thể tách metadata tương ứng

#     # Tách Test In-Domain trước
#     # Kiểm tra xem có đủ mẫu để tách không và test_size có hợp lý không
#     can_split_test = len(y_encoded_all) >= 2 and \
#                      (len(y_encoded_all) * cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO >= 1 or \
#                       cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO == 0) and \
#                      (len(np.unique(y_encoded_all)) >= 2 or not cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO > 0) # Cần ít nhất 2 lớp để stratify

#     if cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO > 0 and can_split_test:
#         try:
#             train_val_indices, test_id_indices, \
#             X_audio_train_val, X_audio_test_id, \
#             X_vcam_train_val, X_vcam_test_id, \
#             y_train_val, y_test_id = train_test_split(
#                 all_indices, X_audio_all, X_vcam_all, y_encoded_all,
#                 test_size=cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO,
#                 random_state=42,
#                 stratify=y_encoded_all
#             )
#             print(f"  Số mẫu Test In-Domain cho fusion: {len(y_test_id)}")
#             if len(y_test_id) > 0: # Chỉ lưu nếu có dữ liệu test
#                 print(f"  Đang lưu dữ liệu features và labels của Test In-Domain...")
#                 os.makedirs(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, exist_ok=True)
#                 with open(cfg.FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH, 'wb') as f: pickle.dump(y_test_id, f)
#                 with open(cfg.FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH, 'wb') as f: pickle.dump(X_audio_test_id, f)
#                 with open(cfg.FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH, 'wb') as f: pickle.dump(X_vcam_test_id, f)

#                 # Lưu metadata cho Test In-Domain
#                 test_id_metadata_df = full_original_metadata_df.iloc[test_id_indices].copy()
#                 test_id_metadata_df.to_csv(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH, index=False)
#                 print(f"    Đã lưu nhãn, features và metadata của Test In-Domain.")
#         except ValueError as e_split_test: # Nếu stratify lỗi do ít mẫu/lớp
#             print(f"CẢNH BÁO: Lỗi khi tách Test In-Domain (có thể do ít mẫu/lớp để stratify): {e_split_test}.")
#             print("  Sử dụng toàn bộ dữ liệu cho Train/Validation, không có Test In-Domain riêng từ bước này.")
#             X_audio_train_val, X_vcam_train_val, y_train_val = X_audio_all, X_vcam_all, y_encoded_all
#             # test_id_indices, X_audio_test_id, X_vcam_test_id, y_test_id sẽ rỗng
#     else:
#         print("  Không tách Test In-Domain (do SPLIT_RATIO=0 hoặc không đủ dữ liệu/lớp). Sử dụng toàn bộ cho Train/Validation.")
#         X_audio_train_val, X_vcam_train_val, y_train_val = X_audio_all, X_vcam_all, y_encoded_all
#         # test_id_indices, X_audio_test_id, X_vcam_test_id, y_test_id sẽ rỗng


#     # Chia tập Train/Validation từ phần còn lại (X_audio_train_val, ...)
#     validation_data_for_fit = None
#     can_split_val = len(y_train_val) >= 2 and \
#                     (len(y_train_val) * cfg.FUSION_VALIDATION_ON_REMAINING_RATIO >= 1 or \
#                      cfg.FUSION_VALIDATION_ON_REMAINING_RATIO == 0) and \
#                     (len(np.unique(y_train_val)) >=2 or not cfg.FUSION_VALIDATION_ON_REMAINING_RATIO > 0)

#     if cfg.FUSION_VALIDATION_ON_REMAINING_RATIO > 0 and can_split_val:
#         try:
#             X_audio_train, X_audio_val, \
#             X_vcam_train, X_vcam_val, \
#             y_train, y_val = train_test_split(
#                 X_audio_train_val, X_vcam_train_val, y_train_val,
#                 test_size=cfg.FUSION_VALIDATION_ON_REMAINING_RATIO,
#                 random_state=42,
#                 stratify=y_train_val
#             )
#             if X_audio_val.size > 0 and y_val.size > 0: # Đảm bảo tập val không rỗng
#                 validation_data_for_fit = ([X_audio_val, X_vcam_val], y_val)
#             else: # Nếu val rỗng sau khi chia (do quá ít mẫu)
#                 print("  Cảnh báo: Tập validation rỗng sau khi chia. Sử dụng toàn bộ X_train_val cho training.")
#                 X_audio_train, X_vcam_train, y_train = X_audio_train_val, X_vcam_train_val, y_train_val
#                 X_audio_val, X_vcam_val, y_val = np.array([]), np.array([]), np.array([]) # Đặt là rỗng
#         except ValueError as e_split_val:
#             print(f"CẢNH BÁO: Lỗi khi tách Validation (có thể do ít mẫu/lớp để stratify): {e_split_val}.")
#             print("  Sử dụng toàn bộ X_train_val cho training, không có Validation set riêng.")
#             X_audio_train, X_vcam_train, y_train = X_audio_train_val, X_vcam_train_val, y_train_val
#             X_audio_val, X_vcam_val, y_val = np.array([]), np.array([]), np.array([])
#     else:
#         print("  Không tách Validation set (do RATIO=0 hoặc không đủ dữ liệu/lớp). Sử dụng toàn bộ X_train_val cho training.")
#         X_audio_train, X_vcam_train, y_train = X_audio_train_val, X_vcam_train_val, y_train_val
#         X_audio_val, X_vcam_val, y_val = np.array([]), np.array([]), np.array([])


#     print(f"  Số mẫu Train cho fusion: {len(y_train)}")
#     if len(y_val) > 0: print(f"  Số mẫu Validation cho fusion: {len(y_val)}")
#     else: print("  Không có tập Validation.")
#     if X_audio_train.size > 0 : print(f"  Shape X_audio_train: {X_audio_train.shape}, X_vcam_train: {X_vcam_train.shape}")


#     # --- 4. Xây dựng Mô hình Fusion ---
#     print("\n--- Xây dựng Mô hình Early Fusion ---")
#     try:
#         # Sử dụng max_len audio từ quá trình fine-tune audio trên YouTube
#         with open(cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP, 'rb') as f:
#             loaded_audio_max_len = pickle.load(f)
#         print(f"  Sử dụng max_len_audio (từ fine-tune audio YT): {loaded_audio_max_len}")
#     except FileNotFoundError:
#         print(f"LỖI: Không tìm thấy file max_len audio: {cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP}.")
#         return

#     audio_input_shape = (loaded_audio_max_len, cfg.AUDIO_N_MFCC)
#     vcam_feature_input_shape = (cfg.FUSION_VCAM_FEATURE_DIM,)

#     # --- 5. Huấn luyện Giai đoạn 1: Đóng băng Nhánh Audio ---
#     print("\n--- Giai đoạn 1: Fine-tuning Fusion với nhánh Audio bị đóng băng ---")
#     # Sử dụng mô hình audio ĐÃ FINE-TUNE TRÊN YOUTUBE để khởi tạo nhánh
#     audio_pretrained_model_for_fusion_path = cfg.AUDIO_MODEL_FOR_FUSION_BRANCH_INIT
#     # Tên lớp nội bộ để trích xuất feature từ mô hình audio (CẦN XÁC MINH CHO MODEL AUDIO ĐÃ FT TRÊN YT)
#     # Giả sử cấu trúc model audio gốc và model audio fine-tune trên YT có cùng tên lớp này
#     # audio_feature_layer_name_for_fusion = 'dropout_3' # Hoặc 'bidirectional_2' nếu bạn lấy output của BiLSTM

#     fusion_model_stage1 = build_early_fusion_model(
#         audio_input_shape=audio_input_shape,
#         vcam_feature_input_shape=vcam_feature_input_shape,
#         num_fusion_classes=cfg.FUSION_NUM_CLASSES,
#         audio_pretrained_model_path=audio_pretrained_model_for_fusion_path,
#         fusion_learning_rate=cfg.FUSION_LEARNING_RATE_STAGE1,
#         audio_branch_trainable=False, # Đóng băng nhánh audio
#         audio_feature_layer_name= cfg.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION
#     )
#     if fusion_model_stage1 is None: print("Lỗi xây dựng model stage 1. Dừng."); return

#     print("Mô hình Fusion Giai đoạn 1 (nhánh Audio đóng băng):")
#     fusion_model_stage1.summary(line_length=150)

#     callbacks_stage1 = [
#         EarlyStopping(monitor='val_loss' if validation_data_for_fit else 'loss', patience=15, restore_best_weights=True, verbose=1), # Tăng patience
#         ModelCheckpoint(cfg.BEST_FUSION_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1),
#         ReduceLROnPlateau(monitor='val_loss' if validation_data_for_fit else 'loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1) # Tăng patience
#     ]

#     print(f"Bắt đầu huấn luyện Giai đoạn 1 với {cfg.FUSION_EPOCHS_STAGE1} epochs...")
#     history_stage1 = fusion_model_stage1.fit(
#         [X_audio_train, X_vcam_train], y_train,
#         validation_data=validation_data_for_fit,
#         epochs=cfg.FUSION_EPOCHS_STAGE1,
#         batch_size=cfg.FUSION_BATCH_SIZE,
#         callbacks=callbacks_stage1,
#         verbose=1
#     )
#     plot_training_history(history_stage1, filename=cfg.FUSION_TRAINING_HISTORY_STAGE1_PLOT_PATH)
#     print(f"Lịch sử huấn luyện Giai đoạn 1 đã lưu: {cfg.FUSION_TRAINING_HISTORY_STAGE1_PLOT_PATH}")

#     # --- 5. Huấn luyện Giai đoạn 2: Mở băng Nhánh Audio (Tùy chọn) ---
#     RUN_STAGE_2_FINETUNING = getattr(cfg, 'RUN_STAGE_2_FINETUNING', True) # Lấy từ config, mặc định True
#     history_stage2 = None # Khởi tạo

#     if RUN_STAGE_2_FINETUNING and cfg.FUSION_EPOCHS_STAGE2 > 0:
#         print("\n--- Giai đoạn 2: Mở băng nhánh Audio và Fine-tuning toàn bộ mô hình Fusion ---")
#         try:
#             # Tải lại model tốt nhất từ Giai đoạn 1
#             fusion_model_stage2 = load_model(cfg.BEST_FUSION_MODEL_SAVE_PATH)
#             print(f"  Đã tải lại model tốt nhất từ Giai đoạn 1: {cfg.BEST_FUSION_MODEL_SAVE_PATH}")
#         except Exception as e_load_s1:
#             print(f"Lỗi khi tải model từ Giai đoạn 1: {e_load_s1}. Sử dụng model hiện tại từ Giai đoạn 1.")
#             fusion_model_stage2 = fusion_model_stage1 # Fallback

#         # Tìm và mở băng nhánh audio
#         try:
#             audio_feature_extractor_submodel = fusion_model_stage2.get_layer('audio_feature_extractor')
#             if audio_feature_extractor_submodel:
#                 audio_feature_extractor_submodel.trainable = True
#                 print("  Nhánh Audio (audio_feature_extractor) đã được mở băng (trainable=True).")
#             else:
#                 print("  CẢNH BÁO: Không tìm thấy layer 'audio_feature_extractor' để mở băng.")
#         except ValueError:
#              print("  CẢNH BÁO: Layer 'audio_feature_extractor' không tồn tại trong model fusion.")


#         fusion_model_stage2.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.FUSION_LEARNING_RATE_STAGE2), # LR rất nhỏ
#             loss=fusion_model_stage2.loss, # Giữ nguyên loss
#             metrics=['accuracy']
#         )
#         print("Mô hình Fusion Giai đoạn 2 (tất cả trainable):")
#         fusion_model_stage2.summary(line_length=150)

#         callbacks_stage2 = [
#             EarlyStopping(monitor='val_loss' if validation_data_for_fit else 'loss', patience=10, restore_best_weights=True, verbose=1), # Có thể giảm patience
#             ModelCheckpoint(cfg.BEST_FUSION_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), # Vẫn lưu model tốt nhất
#             ReduceLROnPlateau(monitor='val_loss' if validation_data_for_fit else 'loss', factor=0.3, patience=5, min_lr=1e-8, verbose=1) # Giảm LR mạnh hơn, min_lr nhỏ hơn
#         ]

#         start_epoch_s2 = 0
#         if history_stage1 and hasattr(history_stage1, 'epoch') and history_stage1.epoch:
#             start_epoch_s2 = len(history_stage1.epoch) # Số epoch đã chạy ở stage 1
        
#         final_epochs_s2_target = cfg.FUSION_EPOCHS_STAGE2 # Số epoch CẦN CHẠY THÊM cho stage 2

#         print(f"Bắt đầu huấn luyện Giai đoạn 2 từ epoch {start_epoch_s2} với {final_epochs_s2_target} epochs (mục tiêu)...")
#         history_stage2 = fusion_model_stage2.fit(
#             [X_audio_train, X_vcam_train], y_train,
#             validation_data=validation_data_for_fit,
#             epochs=start_epoch_s2 + final_epochs_s2_target, # Tổng số epoch từ đầu
#             initial_epoch=start_epoch_s2, # Bắt đầu từ epoch này
#             batch_size=cfg.FUSION_BATCH_SIZE,
#             callbacks=callbacks_stage2,
#             verbose=1
#         )
#         if history_stage2 and history_stage2.history:
#             plot_training_history(history_stage2, filename=cfg.FUSION_TRAINING_HISTORY_STAGE2_PLOT_PATH, initial_epoch_offset=start_epoch_s2)
#             print(f"Lịch sử huấn luyện Giai đoạn 2 đã lưu: {cfg.FUSION_TRAINING_HISTORY_STAGE2_PLOT_PATH}")

#         else:
#             print("Cảnh báo: Không có lịch sử huấn luyện Giai đoạn 2 để vẽ.")
#     else:
#         print("\nBỏ qua Giai đoạn 2 Fine-tune (do RUN_STAGE_2_FINETUNING=False hoặc EPOCHS_STAGE2=0).")


#     # --- 6. Đánh giá cuối cùng trên Tập Validation (của bộ Fine-tune) ---
#     if validation_data_for_fit and y_val.size > 0:
#         print("\n--- Đánh giá Mô hình Fusion Tốt nhất trên Tập VALIDATION (của bộ Fine-tune) ---")
#         try:
#             final_fusion_model = load_model(cfg.BEST_FUSION_MODEL_SAVE_PATH) # Model tốt nhất đã được lưu
#             print(f"  Đã tải model fusion tốt nhất từ: {cfg.BEST_FUSION_MODEL_SAVE_PATH}")
#         except Exception as e_load_final:
#             print(f"Lỗi tải model fusion cuối cùng: {e_load_final}. Dừng đánh giá.")
#             return

#         val_pred_probs_fusion = final_fusion_model.predict([X_audio_val, X_vcam_val])
#         if cfg.FUSION_NUM_CLASSES == 1: # Xử lý cho binary classification nếu cần
#             val_pred_classes_fusion = (val_pred_probs_fusion > 0.5).astype("int32").flatten()
#         else:
#             val_pred_classes_fusion = np.argmax(val_pred_probs_fusion, axis=1)

#         print("\nClassification Report (Fusion Model - Validation Set):")
#         # Sử dụng fusion_label_encoder.classes_ để lấy tên các lớp
#         target_names_for_report = list(fusion_label_encoder.classes_)
#         labels_for_report = range(len(target_names_for_report))
        
#         print(classification_report(y_val, val_pred_classes_fusion,
#                                     target_names=list(fusion_label_encoder.classes_),
#                                     labels=range(len(fusion_label_encoder.classes_)), # Đảm bảo tất cả các lớp được xem xét
#                                     zero_division=0))

#         plot_custom_confusion_matrix(y_val, val_pred_classes_fusion,
#                                      classes=target_names_for_report, # <<< SỬA Ở ĐÂY
#                                      filename=cfg.FUSION_CONFUSION_MATRIX_VAL_PLOT_PATH,
#                                      title="Fusion Validation - ") # Thêm title_prefix nếu muốn
        
#         print(f"Ma trận nhầm lẫn trên tập Validation đã lưu: {cfg.FUSION_CONFUSION_MATRIX_VAL_PLOT_PATH}")
#     else:
#         print("\nKhông có dữ liệu validation để đánh giá mô hình fusion sau huấn luyện.")

#     print("===== EARLY FUSION MODEL FINE-TUNING PIPELINE FINISHED =====")
#     # Thông báo về Test In-Domain (nếu có)
#     # if os.path.exists(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH):
#     #     print(f"Tập Test In-Domain (features, labels, và metadata) đã được lưu.")
#     #     print(f"  Sẵn sàng để đánh giá riêng biệt bằng các script evaluation.")
#     # else:
#     #     print("Không có tập Test In-Domain nào được tạo ra từ pipeline này.")


# if __name__ == '__main__':
#     # Gọi ensure_output_directories ở đầu main_finetune_fusion.py nếu nó không tự chạy khi import config
#     # if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories):
#     #     print("Đảm bảo các thư mục output tồn tại (từ main_finetune_fusion.py)...")
#     #     cfg.ensure_output_directories()

#     required_for_fusion_ft_pipe = [
#         cfg.AUDIO_MODEL_FOR_FUSION_BRANCH_INIT,
#         cfg.AUDIO_SCALER_FOR_FUSION_PREP,
#         cfg.AUDIO_MAX_LEN_FOR_FUSION_PREP,
#         cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION,
#         cfg.FINETUNE_MULTICLASS_METADATA_FILE
#     ]
#     missing_reqs_fusion = [f for f in required_for_fusion_ft_pipe if not os.path.exists(f)]
#     if missing_reqs_fusion:
#         print("LỖI: Thiếu các file đầu vào quan trọng cho pipeline fine-tuning fusion:")
#         for f_path in missing_reqs_fusion: print(f" - {f_path}")
#     else:
#         run_fusion_finetuning_pipeline()


# Drone_Detection_Project/src/training/fine_tune_fusion_pipeline.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model # Giữ Model
from tensorflow.keras.layers import Dense # Đảm bảo import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd

try:
    from src.config_loader.loader import get_config
    from src.model_architectures.fusion_model import build_early_fusion_model
    from src.evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix
except ImportError:
    import sys
    current_dir_ft_pipe = os.path.dirname(os.path.abspath(__file__))
    src_dir_ft_pipe = os.path.dirname(current_dir_ft_pipe)
    project_root_ft_pipe = os.path.dirname(src_dir_ft_pipe)
    if project_root_ft_pipe not in sys.path: sys.path.insert(0, project_root_ft_pipe)
    if src_dir_ft_pipe not in sys.path: sys.path.insert(0, src_dir_ft_pipe)
    from config_loader.loader import get_config
    from model_architectures.fusion_model import build_early_fusion_model
    from evaluation.plotting_utils import plot_training_history, plot_custom_confusion_matrix

cfg = get_config()

def load_fine_tune_data_for_fusion_pipeline(metadata_file_path, paired_data_base_dir):
    print(f"--- Đang tải dữ liệu fine-tune đã xử lý cho FUSION từ metadata: {metadata_file_path} ---")
    if not os.path.exists(metadata_file_path):
        print(f"LỖI: File metadata không tồn tại: {metadata_file_path}")
        return None, None, None, None
    try:
        full_original_metadata_df = pd.read_csv(metadata_file_path)
    except Exception as e:
        print(f"Lỗi khi đọc file metadata: {e}")
        return None, None, None, None

    all_audio_features, all_vcam_features, all_labels_text = [], [], []
    original_indices_for_split = [] # Để theo dõi index gốc cho việc tách metadata

    for index, row in tqdm(full_original_metadata_df.iterrows(), total=full_original_metadata_df.shape[0], desc="Loading paired features for fusion"):
        try:
            audio_mfcc_rel_path = str(row['audio_mfcc_file']).strip()
            vcam_features_rel_path = str(row['vcam_features_file']).strip()
            label = str(row['label']).strip()

            audio_mfcc_abs_path = os.path.join(paired_data_base_dir, audio_mfcc_rel_path)
            vcam_features_abs_path = os.path.join(paired_data_base_dir, vcam_features_rel_path)

            if not os.path.exists(audio_mfcc_abs_path) or not os.path.exists(vcam_features_abs_path):
                continue

            audio_feat = np.load(audio_mfcc_abs_path)
            vcam_feat = np.load(vcam_features_abs_path)

            # Kiểm tra shape cơ bản (chỉ số chiều cuối của audio và chiều đầu của vcam)
            if audio_feat.shape[-1] != cfg.AUDIO_N_MFCC:
                print(f"Cảnh báo: Audio feature shape không đúng ({audio_feat.shape}, mong đợi *x{cfg.AUDIO_N_MFCC}) cho {audio_mfcc_abs_path}")
                continue
            if vcam_feat.shape[0] != cfg.FUSION_VCAM_FEATURE_DIM:
                print(f"Cảnh báo: VCam feature shape không đúng ({vcam_feat.shape}, mong đợi ({cfg.FUSION_VCAM_FEATURE_DIM},)) cho {vcam_features_abs_path}")
                continue
            
            # Đảm bảo audio_feat có 3 chiều (batch_size_implicit, timesteps, features)
            # Vì khi lưu np.save(audio_mfcc_save_path, mfcc_s[0]), mfcc_s[0] đã là (timesteps, features)
            # Nên khi load lại, nó vẫn là (timesteps, features). np.array(all_audio_features) sẽ tạo (N, timesteps, features)
            if audio_feat.ndim != 2 : # Mong đợi (timesteps, features)
                print(f"Cảnh báo: Audio feature ndim không đúng ({audio_feat.ndim}, mong đợi 2) cho {audio_mfcc_abs_path}")
                continue


            all_audio_features.append(audio_feat)
            all_vcam_features.append(vcam_feat)
            all_labels_text.append(label)
            original_indices_for_split.append(index) # Lưu index của dòng metadata gốc
        except Exception as e:
            print(f"Lỗi khi tải features cho segment {row.get('segment_id', f'index {index}')}: {e}")
            continue

    if not all_labels_text:
        print("Không tải được mẫu dữ liệu fine-tune nào cho fusion.")
        return None, None, None, None

    print(f"Đã tải thành công {len(all_labels_text)} cặp đặc trưng (Audio MFCC, VCam) cho fusion.")
    # Chuyển thành numpy array ở đây
    return np.array(all_audio_features), np.array(all_vcam_features), all_labels_text, \
           full_original_metadata_df, np.array(original_indices_for_split)


def run_fusion_finetuning_pipeline():
    print("===== STARTING EARLY FUSION MODEL FINE-TUNING PIPELINE (VCam + Audio) =====")
    if hasattr(cfg, 'ensure_output_directories') and callable(cfg.ensure_output_directories):
         cfg.ensure_output_directories()

    X_audio_all, X_vcam_all, y_labels_text_all, full_original_metadata_df, original_indices_all = \
        load_fine_tune_data_for_fusion_pipeline(
            metadata_file_path=cfg.FINETUNE_MULTICLASS_METADATA_FILE,
            paired_data_base_dir=cfg.FINETUNE_PAIRED_DATA_BASE_DIR
        )

    if X_audio_all is None or X_audio_all.size == 0 or not y_labels_text_all:
        print("Không có dữ liệu fine-tune để huấn luyện. Dừng pipeline.")
        return

    print(f"\n--- Mã hóa Nhãn cho Fusion ({len(cfg.FUSION_CLASS_NAMES)} lớp) ---")
    fusion_label_encoder = LabelEncoder()
    try:
        fusion_label_encoder.fit(cfg.FUSION_CLASS_NAMES)
        y_encoded_all = fusion_label_encoder.transform(y_labels_text_all)
        print(f"  Các lớp trong Fusion LabelEncoder: {list(fusion_label_encoder.classes_)}")
    except ValueError as e: # ... (Xử lý lỗi)
        print(f"LỖI Mã hóa Nhãn cho Fusion: {e}...")
        return
    print(f"Đang lưu Fusion LabelEncoder vào: {cfg.FUSION_LABEL_ENCODER_PATH}")
    os.makedirs(os.path.dirname(cfg.FUSION_LABEL_ENCODER_PATH), exist_ok=True)
    with open(cfg.FUSION_LABEL_ENCODER_PATH, 'wb') as f: pickle.dump(fusion_label_encoder, f)


    print("\n--- Chia dữ liệu fine-tune thành Train/Validation và Test In-Domain ---")
    # ... (Logic chia Train/Val và Test In-Domain như bạn đã cung cấp ở Câu hỏi 38) ...
    # ... (Sử dụng cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO và cfg.FUSION_VALIDATION_ON_REMAINING_RATIO) ...
    # ... (Lưu Test In-Domain features, labels và metadata) ...
    # Đoạn code chia dữ liệu từ Câu hỏi 38 (đã được sửa đổi và kiểm tra):
    X_audio_train_val, X_vcam_train_val, y_train_val = X_audio_all, X_vcam_all, y_encoded_all
    indices_train_val_metadata = original_indices_all # Ban đầu, tất cả là train_val
    
    X_audio_test_id, X_vcam_test_id, y_test_id = np.array([]), np.array([]), np.array([]) # Khởi tạo rỗng

    can_split_test_fusion = len(y_encoded_all) >= 2 and \
                           (len(y_encoded_all) * cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO >= 1 or cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO == 0) and \
                           (len(y_encoded_all) * (1-cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO) >= 1 or (1-cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO) == 0) and \
                           (len(np.unique(y_encoded_all)) >= 2 or cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO == 0 or (1-cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO) == 0)

    if cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO > 0 and can_split_test_fusion:
        try:
            indices_train_val_metadata, indices_test_id_metadata, \
            X_audio_train_val, X_audio_test_id, \
            X_vcam_train_val, X_vcam_test_id, \
            y_train_val, y_test_id = train_test_split(
                original_indices_all, X_audio_all, X_vcam_all, y_encoded_all,
                test_size=cfg.FUSION_TEST_IN_DOMAIN_SPLIT_RATIO,
                random_state=42, stratify=y_encoded_all
            )
            print(f"  Số mẫu Test In-Domain cho fusion: {len(y_test_id)}")
            if len(y_test_id) > 0:
                print(f"  Đang lưu dữ liệu features và labels của Test In-Domain cho Fusion...")
                os.makedirs(cfg.FUSION_TEST_IN_DOMAIN_DATA_DIR, exist_ok=True)
                with open(cfg.FUSION_TEST_IN_DOMAIN_TRUE_LABELS_PATH, 'wb') as f: pickle.dump(y_test_id, f)
                with open(cfg.FUSION_TEST_IN_DOMAIN_AUDIO_FEATURES_PATH, 'wb') as f: pickle.dump(X_audio_test_id, f)
                with open(cfg.FUSION_TEST_IN_DOMAIN_VCAM_FEATURES_PATH, 'wb') as f: pickle.dump(X_vcam_test_id, f)
                if full_original_metadata_df is not None and indices_test_id_metadata.size > 0:
                    test_id_metadata_df = full_original_metadata_df.iloc[indices_test_id_metadata].copy()
                    test_id_metadata_df.to_csv(cfg.FUSION_TEST_IN_DOMAIN_METADATA_PATH, index=False)
                print(f"    Đã lưu nhãn, features và metadata của Test In-Domain cho Fusion.")
        except ValueError as e_split_test:
            print(f"CẢNH BÁO: Lỗi khi tách Test In-Domain cho Fusion: {e_split_test}. Sử dụng toàn bộ cho Train/Val.")
            # Giữ nguyên giá trị ban đầu
    else:
        print("  Không tách Test In-Domain cho Fusion. Sử dụng toàn bộ cho Train/Validation.")

    # Chia Train/Validation từ phần còn lại
    validation_data_for_fit = None
    X_audio_train, X_vcam_train, y_train = X_audio_train_val, X_vcam_train_val, y_train_val # Mặc định

    can_split_val_fusion = len(y_train_val) >= 2 and \
                           (len(y_train_val) * cfg.FUSION_VALIDATION_ON_REMAINING_RATIO >= 1 or cfg.FUSION_VALIDATION_ON_REMAINING_RATIO == 0) and \
                           (len(y_train_val) * (1-cfg.FUSION_VALIDATION_ON_REMAINING_RATIO) >=1 or (1-cfg.FUSION_VALIDATION_ON_REMAINING_RATIO) == 0) and \
                           (len(np.unique(y_train_val)) >=2 or cfg.FUSION_VALIDATION_ON_REMAINING_RATIO == 0 or (1-cfg.FUSION_VALIDATION_ON_REMAINING_RATIO)==0)

    if cfg.FUSION_VALIDATION_ON_REMAINING_RATIO > 0 and can_split_val_fusion:
        try:
            X_audio_train, X_audio_val, \
            X_vcam_train, X_vcam_val, \
            y_train, y_val = train_test_split(
                X_audio_train_val, X_vcam_train_val, y_train_val,
                test_size=cfg.FUSION_VALIDATION_ON_REMAINING_RATIO,
                random_state=42, stratify=y_train_val
            )
            if X_audio_val.size > 0 and y_val.size > 0:
                validation_data_for_fit = ([X_audio_val, X_vcam_val], y_val)
            else:
                print("  Cảnh báo: Tập validation fusion rỗng sau khi chia.")
        except ValueError as e_split_val:
            print(f"CẢNH BÁO: Lỗi khi tách Validation cho Fusion: {e_split_val}.")
    else:
        print("  Không tách Validation set cho Fusion.")

    print(f"  Số mẫu Train cho fusion: {len(y_train)}")
    if validation_data_for_fit: print(f"  Số mẫu Validation cho fusion: {len(y_val)}")
    else: print("  Không có tập Validation cho fusion.")
    if X_audio_train.size > 0 : print(f"  Shape X_audio_train: {X_audio_train.shape}, X_vcam_train: {X_vcam_train.shape}")
    else: print("LỖI: Tập training fusion rỗng!"); return


    # --- 4. Xây dựng Mô hình Fusion ---
    print("\n--- Xây dựng Mô hình Early Fusion ---")
    try:
        with open(cfg.AUDIO_YOUTUBE_MAX_LEN_PATH, 'rb') as f: # Dùng max_len từ audio YouTube fine-tune
            loaded_audio_max_len_for_fusion = pickle.load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file max_len audio: {cfg.AUDIO_YOUTUBE_MAX_LEN_PATH}.")
        return

    audio_input_shape = (loaded_audio_max_len_for_fusion, cfg.AUDIO_N_MFCC)
    vcam_feature_input_shape = (cfg.FUSION_VCAM_FEATURE_DIM,)

    # --- 5. Huấn luyện Giai đoạn 1: Đóng băng Nhánh Audio ---
    print("\n--- Giai đoạn 1: Fine-tuning Fusion với nhánh Audio bị đóng băng ---")
    # audio_pretrained_model_for_fusion_path phải là model audio ĐÃ FINE-TUNE trên YouTube
    audio_pretrained_model_for_fusion_path = cfg.AUDIO_MODEL_FOR_FUSION_BRANCH_INIT
    # audio_feature_layer_name_for_fusion là tên lớp trong model audio ĐÃ FINE-TUNE ở trên
    audio_feature_layer_name_for_fusion = cfg.AUDIO_FEATURE_LAYER_NAME_FOR_FUSION

    fusion_model_stage1 = build_early_fusion_model(
        audio_input_shape=audio_input_shape,
        vcam_feature_input_shape=vcam_feature_input_shape,
        num_fusion_classes=cfg.FUSION_NUM_CLASSES,
        audio_pretrained_model_path=audio_pretrained_model_for_fusion_path,
        fusion_learning_rate=cfg.FUSION_LEARNING_RATE_STAGE1,
        audio_branch_trainable=False,
        audio_feature_layer_name=audio_feature_layer_name_for_fusion
    )
    if fusion_model_stage1 is None: print("Lỗi xây dựng model stage 1. Dừng."); return
    print("Mô hình Fusion Giai đoạn 1 (nhánh Audio đóng băng):")
    fusion_model_stage1.summary(line_length=150)
    callbacks_stage1 = [ EarlyStopping(monitor='val_loss' if validation_data_for_fit else 'loss', patience=15, restore_best_weights=True, verbose=1), ModelCheckpoint(cfg.BEST_FUSION_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), ReduceLROnPlateau(monitor='val_loss' if validation_data_for_fit else 'loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1)]
    print(f"Bắt đầu huấn luyện Giai đoạn 1 với {cfg.FUSION_EPOCHS_STAGE1} epochs...")
    history_stage1 = fusion_model_stage1.fit( [X_audio_train, X_vcam_train], y_train, validation_data=validation_data_for_fit, epochs=cfg.FUSION_EPOCHS_STAGE1, batch_size=cfg.FUSION_BATCH_SIZE, callbacks=callbacks_stage1, verbose=1)
    plot_training_history(history_stage1, filename=cfg.FUSION_TRAINING_HISTORY_STAGE1_PLOT_PATH, title_prefix="Fusion Stage 1 - ")

    # --- 6. Huấn luyện Giai đoạn 2: Mở băng Nhánh Audio (Tùy chọn) ---
    # ... (Giữ nguyên logic huấn luyện Giai đoạn 2 từ Câu hỏi 38, đảm bảo các biến được tham chiếu đúng) ...
    # ... (bao gồm tải lại model, mở băng nhánh audio, compile lại, và fit) ...
    history_stage2 = None
    RUN_STAGE_2_FINETUNING = getattr(cfg, 'RUN_STAGE_2_FINETUNING', True)
    if RUN_STAGE_2_FINETUNING and cfg.FUSION_EPOCHS_STAGE2 > 0:
        print("\n--- Giai đoạn 2: Mở băng nhánh Audio và Fine-tuning toàn bộ mô hình Fusion ---")
        try:
            fusion_model_stage2 = load_model(cfg.BEST_FUSION_MODEL_SAVE_PATH) # Tải model tốt nhất từ Stage 1
        except Exception as e_load_s1:
            print(f"Lỗi khi tải model từ Giai đoạn 1: {e_load_s1}. Sử dụng model hiện tại từ Giai đoạn 1."); fusion_model_stage2 = fusion_model_stage1
        try:
            audio_feature_extractor_submodel = fusion_model_stage2.get_layer('audio_feature_extractor')
            if audio_feature_extractor_submodel: audio_feature_extractor_submodel.trainable = True; print("  Nhánh Audio đã mở băng.")
            else: print("  CẢNH BÁO: Không tìm thấy 'audio_feature_extractor' để mở băng.")
        except ValueError: print("  CẢNH BÁO: Layer 'audio_feature_extractor' không tồn tại trong model fusion.")
        fusion_model_stage2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.FUSION_LEARNING_RATE_STAGE2), loss=fusion_model_stage2.loss, metrics=['accuracy'])
        print("Mô hình Fusion Giai đoạn 2 (tất cả trainable):"); fusion_model_stage2.summary(line_length=150)
        callbacks_stage2 = [EarlyStopping(monitor='val_loss' if validation_data_for_fit else 'loss', patience=10, restore_best_weights=True, verbose=1), ModelCheckpoint(cfg.BEST_FUSION_MODEL_SAVE_PATH, monitor='val_accuracy' if validation_data_for_fit else 'accuracy', save_best_only=True, mode='max', verbose=1), ReduceLROnPlateau(monitor='val_loss' if validation_data_for_fit else 'loss', factor=0.3, patience=5, min_lr=1e-8, verbose=1)]
        start_epoch_s2 = 0
        if history_stage1 and hasattr(history_stage1, 'epoch') and history_stage1.epoch: start_epoch_s2 = len(history_stage1.epoch)
        final_epochs_s2_target = cfg.FUSION_EPOCHS_STAGE2
        print(f"Bắt đầu huấn luyện Giai đoạn 2 từ epoch {start_epoch_s2} với {final_epochs_s2_target} epochs (mục tiêu)...")
        history_stage2 = fusion_model_stage2.fit([X_audio_train, X_vcam_train], y_train, validation_data=validation_data_for_fit, epochs=start_epoch_s2 + final_epochs_s2_target, initial_epoch=start_epoch_s2, batch_size=cfg.FUSION_BATCH_SIZE, callbacks=callbacks_stage2, verbose=1)
        if history_stage2 and history_stage2.history: plot_training_history(history_stage2, filename=cfg.FUSION_TRAINING_HISTORY_STAGE2_PLOT_PATH, title_prefix="Fusion Stage 2 - ", initial_epoch_offset=start_epoch_s2)
    else: print("\nBỏ qua Giai đoạn 2 Fine-tune Fusion.")


    # --- 7. Đánh giá cuối cùng trên Tập Validation (của bộ Fine-tune Fusion) ---
    # ... (Giữ nguyên logic đánh giá trên tập Val của Fusion từ Câu hỏi 38) ...
    # ... (Bao gồm tải model tốt nhất, predict, classification_report, plot_custom_confusion_matrix) ...
    if validation_data_for_fit and y_val.size > 0: # y_val giờ là y_val của Fusion
        print("\n--- Đánh giá Mô hình Fusion Tốt nhất trên Tập VALIDATION (của bộ Fine-tune Fusion) ---")
        try: final_fusion_model = load_model(cfg.BEST_FUSION_MODEL_SAVE_PATH)
        except Exception as e_load_final: print(f"Lỗi tải model fusion cuối cùng: {e_load_final}. Dừng đánh giá."); return
        val_pred_probs_fusion = final_fusion_model.predict([X_audio_val, X_vcam_val]) # X_audio_val, X_vcam_val của Fusion
        if cfg.FUSION_NUM_CLASSES == 1: val_pred_classes_fusion = (val_pred_probs_fusion > 0.5).astype("int32").flatten()
        else: val_pred_classes_fusion = np.argmax(val_pred_probs_fusion, axis=1)
        print("\nClassification Report (Fusion Model - Validation Set):")
        target_names_fusion_report = list(fusion_label_encoder.classes_)
        labels_fusion_report = range(len(target_names_fusion_report))
        report_fusion_val_str = classification_report(y_val, val_pred_classes_fusion, target_names=target_names_fusion_report, labels=labels_fusion_report, zero_division=0)
        print(report_fusion_val_str)
        report_fusion_val_path = os.path.join(cfg.FUSION_REPORTS_METRICS_DIR, "fusion_classification_report_val.txt") # Đổi tên file
        with open(report_fusion_val_path, 'w') as f: f.write(report_fusion_val_str)
        print(f"Đã lưu Classification Report (Validation Fusion) vào: {report_fusion_val_path}")
        plot_custom_confusion_matrix(y_val, val_pred_classes_fusion, classes=target_names_fusion_report, filename=cfg.FUSION_CONFUSION_MATRIX_VAL_PLOT_PATH, title="Fusion Validation - ")
    else: print("\nKhông có dữ liệu validation để đánh giá mô hình fusion sau huấn luyện.")

    print("===== EARLY FUSION MODEL FINE-TUNING PIPELINE FINISHED =====")
    if X_audio_test_id.size > 0: # Kiểm tra xem tập test có được tạo không
        print(f"\nTập Test In-Domain cho Fusion (features, labels, và metadata) đã được lưu.")
    else:
        print("\nKhông có tập Test In-Domain nào được tạo ra cho Fusion từ pipeline này.")


if __name__ == '__main__':
    required_for_fusion_ft_pipe = [
        cfg.AUDIO_MODEL_FOR_FUSION_BRANCH_INIT,
        cfg.AUDIO_YOUTUBE_SCALER_PATH, # Đổi thành AUDIO_YOUTUBE_SCALER_PATH
        cfg.AUDIO_YOUTUBE_MAX_LEN_PATH,  # Đổi thành AUDIO_YOUTUBE_MAX_LEN_PATH
        cfg.VCAM_MODEL_FOR_FUSION_FEATURE_EXTRACTION,
        cfg.FINETUNE_MULTICLASS_METADATA_FILE
    ]
    missing_reqs_fusion = [f for f in required_for_fusion_ft_pipe if not os.path.exists(f)]
    if missing_reqs_fusion:
        print("LỖI: Thiếu các file đầu vào quan trọng cho pipeline fine-tuning fusion:")
        for f_path in missing_reqs_fusion: print(f" - {f_path}")
    else:
        run_fusion_finetuning_pipeline()