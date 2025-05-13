# Drone_Detection_Project/src/data_processing/audio_utils.py
import librosa
import numpy as np
import os
import glob
import random # Cần cho load_select_split_process_data
# KHÔNG cần import từ các module anh em trong file này (nếu chỉ chứa các hàm tiện ích cơ bản)

def extract_mfcc(audio_segment, sr, n_mfcc, n_fft, hop_length):
    # ... (code hàm extract_mfcc như bạn cung cấp) ...
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length,
                                 window='hamming', center=False)
    return mfccs.T

def pad_features(features, max_len=None):
    # ... (code hàm pad_features như bạn cung cấp) ...
    if not features: # Kiểm tra nếu list rỗng
        return np.array([]), 0
    # Lọc ra các feature hợp lệ trước khi tìm max_len
    valid_features = [feat for feat in features if isinstance(feat, np.ndarray) and feat.ndim > 0 and feat.shape[0] > 0]
    if not valid_features:
         return np.array([]), 0 # Trả về array rỗng và max_len = 0 nếu không có feature hợp lệ

    if max_len is None:
        max_len = max(feat.shape[0] for feat in valid_features) # Tìm max_len an toàn hơn
    if max_len == 0: # Nếu không có feature hợp lệ nào có độ dài > 0
        return np.array([]), 0

    padded_features_list = []
    for feat in features: # Lặp lại qua features gốc để giữ thứ tự và xử lý lỗi
         if isinstance(feat, np.ndarray) and feat.ndim > 0 and feat.shape[0] > 0: # Bỏ qua feature lỗi hoặc rỗng
            if feat.shape[0] < max_len:
                padded = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant')
            else:
                padded = feat[:max_len, :]
            padded_features_list.append(padded)
         # else: có thể thêm log cảnh báo về feature không hợp lệ

    if not padded_features_list: # Nếu không có feature nào được đệm
         return np.array([]), max_len

    padded_features = np.array(padded_features_list)
    # print(f"Features padded to shape: {padded_features.shape}") # Bỏ print ở hàm con
    return padded_features, max_len


def scale_features(features_padded, scaler=None):
    # ... (code hàm scale_features như bạn cung cấp, đảm bảo scaler được import từ sklearn) ...
    from sklearn.preprocessing import StandardScaler # Import ở đây nếu chỉ dùng trong hàm này
    if not isinstance(features_padded, np.ndarray) or features_padded.size == 0:
        # print("Warning: Cannot scale empty or invalid feature array.") # Bỏ print
        return features_padded, scaler if scaler else StandardScaler()
    # ... (phần còn lại của hàm scale_features)
    original_shape = features_padded.shape
    if features_padded.ndim != 3:
        #  print(f"Warning: Expected 3D array for scaling, got {features_padded.ndim}D. Returning original features.")
         return features_padded, scaler if scaler else StandardScaler()

    n_samples, n_timesteps, n_features = original_shape
    if n_features == 0:
        #  print("Warning: Feature dimension is zero. Cannot scale. Returning original features.")
         return features_padded, scaler if scaler else StandardScaler()

    features_reshaped = features_padded.reshape(-1, n_features)

    if scaler is None:
        scaler = StandardScaler()
        if features_reshaped.shape[0] > 0:
            try:
                scaler.fit(features_reshaped)
            except ValueError as e:
                #  print(f"Error fitting scaler: {e}. Returning original features.")
                 return features_padded, scaler
        # else:
            # print("Warning: No data to fit the scaler.")

    features_scaled = features_padded # Default to original if scaling fails
    if hasattr(scaler, 'mean_') and scaler.mean_ is not None and features_reshaped.shape[0] > 0:
        try:
            features_scaled_reshaped = scaler.transform(features_reshaped)
            features_scaled = features_scaled_reshaped.reshape(original_shape)
            # print("Features scaled using StandardScaler.")
        except ValueError as e:
            # print(f"Error transforming features: {e}. Returning original features.")
            features_scaled = features_padded # Revert to original on error
        except Exception as e:
            # print(f"An unexpected error occurred during scaling: {e}. Returning original features.")
            features_scaled = features_padded # Revert to original on error
    # else:
        # print("Warning: Scaling could not be performed (scaler not fitted or no data). Returning original features.")

    return features_scaled, scaler


def check_directory_structure(base_path, classes, num_test_files_per_class): # Thêm num_test_files_per_class
    # ... (code hàm check_directory_structure như bạn cung cấp) ...
    print(f"--- Checking Directory Structure at: {base_path} ---")
    if not os.path.exists(base_path):
        print(f"ERROR: Base path not found: {base_path}")
        return False
    all_classes_found = True
    min_files_for_test = num_test_files_per_class
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        if os.path.exists(class_path) and os.path.isdir(class_path):
            wav_files = glob.glob(os.path.join(class_path, '*.wav'))
            print(f"Class '{class_name}': Found {len(wav_files)} .wav files.")
            if len(wav_files) < min_files_for_test:
                 print(f"WARNING: Class '{class_name}' has fewer files ({len(wav_files)}) than required for test set ({min_files_for_test}).")
            if len(wav_files) == 0:
                 print(f"WARNING: No .wav files found in {class_path}")
        else:
            print(f"ERROR: Class directory not found or not a directory: {class_path}")
            all_classes_found = False
    # print("-" * 50) # Bỏ print ở hàm con
    return all_classes_found


def load_select_split_process_data(base_path, classes, segment_duration, sr,
                                  n_mfcc, n_fft, hop_length, # Thêm các tham số MFCC
                                  num_test_files_per_class,
                                  test_files_list_path):
    # ... (code hàm load_select_split_process_data như bạn cung cấp) ...
    # Đảm bảo hàm extract_mfcc được gọi đúng với các tham số n_mfcc, n_fft, hop_length
    from sklearn.preprocessing import LabelEncoder # Import ở đây nếu chỉ dùng trong hàm này
    print("--- Loading, Selecting Test Files, and Processing Data ---")
    all_files_by_class = {cls: glob.glob(os.path.join(base_path, cls, '*.wav')) for cls in classes}

    test_files_dict = {}
    train_val_files_dict = {}
    selected_test_filenames = []

    print("Selecting test files randomly...")
    for class_name, file_list in all_files_by_class.items():
        if len(file_list) < num_test_files_per_class:
            print(f"Warning: Not enough files in class '{class_name}' for test set ({num_test_files_per_class} required). Using {len(file_list)} files as test.")
            test_files_dict[class_name] = list(file_list) # Chuyển sang list
            train_val_files_dict[class_name] = []
        else:
            local_file_list = list(file_list) # Tạo bản copy để shuffle
            random.shuffle(local_file_list)
            test_files_dict[class_name] = local_file_list[:num_test_files_per_class]
            train_val_files_dict[class_name] = local_file_list[num_test_files_per_class:]

        selected_test_filenames.extend([os.path.basename(f) for f in test_files_dict[class_name]])
        print(f"  Class '{class_name}': Selected {len(test_files_dict[class_name])} test files, {len(train_val_files_dict[class_name])} for train/val.")


    try:
        os.makedirs(os.path.dirname(test_files_list_path), exist_ok=True) # Đảm bảo thư mục tồn tại
        with open(test_files_list_path, 'w') as f:
            for fname in sorted(selected_test_filenames):
                f.write(fname + '\n')
        print(f"Test filenames saved to: {test_files_list_path}")
    except Exception as e:
        print(f"Error saving test file list: {e}")

    print("Processing train/validation files...")
    train_val_features = []
    train_val_labels = []
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    for class_name, file_list_for_class in train_val_files_dict.items(): # Đổi tên biến
        print(f"  Processing train/val for class '{class_name}' ({len(file_list_for_class)} files)...")
        for file_path in file_list_for_class:
            try:
                audio, current_sr_load = librosa.load(file_path, sr=sr) # Đảm bảo load với sr mong muốn
                if current_sr_load != sr:
                    print(f"    Warning: File {file_path} loaded with sr {current_sr_load}, but model expects {sr}. Resampling occurred.")

                segment_samples = int(segment_duration * sr)
                for start in range(0, len(audio) - segment_samples + 1, segment_samples):
                    segment = audio[start : start + segment_samples]
                    if len(segment) == segment_samples:
                        # Gọi extract_mfcc với đầy đủ tham số
                        mfccs = extract_mfcc(segment, sr, n_mfcc, n_fft, hop_length)
                        train_val_features.append(mfccs)
                        train_val_labels.append(class_name)
            except Exception as e:
                print(f"  Warning: Could not process file {file_path}: {e}")

    if not train_val_features:
        # raise ValueError("No train/validation features extracted. Check remaining audio files.")
        print("ERROR: No train/validation features extracted. Check remaining audio files. Exiting.")
        return [], np.array([]), label_encoder, [] # Trả về giá trị rỗng để pipeline có thể dừng

    encoded_train_val_labels = label_encoder.transform(train_val_labels)

    print(f"Total train/validation segments processed: {len(train_val_features)}")
    # print("-" * 50) # Bỏ print ở hàm con
    return train_val_features, np.array(encoded_train_val_labels), label_encoder, selected_test_filenames