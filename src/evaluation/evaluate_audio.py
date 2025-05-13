# Drone_Detection_Project/src/evaluation/evaluate_audio.py
import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, classification_report

# SỬA IMPORT: Sử dụng import tuyệt đối từ package src
from src.data_processing.audio_utils import extract_mfcc, pad_features, scale_features
from src.evaluation.plotting_utils import plot_custom_confusion_matrix

def test_audio_model_on_saved_files(
        base_path, # Đường dẫn đến thư mục chứa các lớp audio thô (ví dụ: data/raw/Audio_Original)
        model_path, label_encoder_path, scaler_path, max_len_path, test_files_list_path,
        # Các tham số cấu hình cần thiết cho xử lý audio trong test
        segment_duration, sr, n_mfcc, n_fft, hop_length,
        # Đường dẫn lưu kết quả
        results_save_path,
        predictions_save_path, true_labels_save_path, class_names_save_path,
        confusion_matrix_save_path):
    # ... (Nội dung hàm test_audio_model_on_saved_files như bạn đã cung cấp ở câu hỏi trước) ...
    # Đảm bảo nó sử dụng các tham số truyền vào cho đường dẫn và cấu hình.
    # Và gọi đúng các hàm đã import từ src.data_processing và src.evaluation
    print("\n--- Testing Audio Model on Selected Test Files ---")

    # 1. Tải các thành phần cần thiết
    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print(f"Loading label encoder from: {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loading max_len from: {max_len_path}")
        with open(max_len_path, 'rb') as f:
            max_len = pickle.load(f)
        print(f"Loading test file list from: {test_files_list_path}")
        with open(test_files_list_path, 'r') as f:
            test_filenames = [line.strip() for line in f if line.strip()]
        print("Model, encoder, scaler, max_len, and test file list loaded successfully.")
        actual_classes = list(label_encoder.classes_)
    except FileNotFoundError as e:
        print(f"ERROR: Could not load required file: {e}. Testing aborted.")
        return
    except Exception as e:
        print(f"ERROR loading components: {e}. Testing aborted.")
        return

    test_file_paths_map = {}
    test_true_labels_map = {}

    for fname in test_filenames:
        found = False
        for class_name_iter in actual_classes:
            fpath = os.path.join(base_path, class_name_iter, fname) # base_path là thư mục chứa các lớp (Audio_Original)
            if os.path.exists(fpath):
                test_file_paths_map[fname] = fpath
                test_true_labels_map[fname] = class_name_iter
                found = True
                break
        if not found:
            print(f"Warning: Test file '{fname}' listed in {test_files_list_path} but not found in any class directory under {base_path}")

    all_true_labels_encoded_list = []
    all_pred_labels_encoded_list = []
    detailed_results = []

    print(f"Processing {len(test_file_paths_map)} found test files...")
    for fname, fpath in test_file_paths_map.items():
        true_label_name = test_true_labels_map.get(fname)
        if true_label_name is None: continue

        try:
            true_label_encoded = label_encoder.transform([true_label_name])[0]
        except ValueError:
            print(f"  Warning: Label '{true_label_name}' for file {fname} not found in LabelEncoder. Skipping.")
            continue
        print(f"  Testing file: {fname} (True Label: {true_label_name})")

        try:
            audio, current_sr_load = librosa.load(fpath, sr=sr)
            if current_sr_load != sr:
                 print(f"    Warning: File {fname} loaded with sr {current_sr_load}, but model expects {sr}.")

            segment_samples = int(segment_duration * sr)
            segment_features_list = []

            for start_sample in range(0, len(audio) - segment_samples + 1, segment_samples):
                segment = audio[start_sample : start_sample + segment_samples]
                if len(segment) == segment_samples:
                    mfccs = extract_mfcc(segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    segment_features_list.append(mfccs)

            if not segment_features_list:
                print(f"  Warning: No valid segments extracted from {fname}. Skipping.")
                continue

            features_padded, _ = pad_features(segment_features_list, max_len=max_len)
            if features_padded.size == 0:
                 print(f"  Warning: No valid features after padding for {fname}. Skipping.")
                 continue
            features_scaled, _ = scale_features(features_padded, scaler=scaler)

            segment_pred_probs = model.predict(features_scaled, verbose=0)
            avg_probs = np.mean(segment_pred_probs, axis=0)
            pred_label_encoded = np.argmax(avg_probs)
            pred_label_name = label_encoder.inverse_transform([pred_label_encoded])[0]
            confidence = avg_probs[pred_label_encoded]

            # print(f"    Predicted: {pred_label_name} (Confidence: {confidence:.4f})") # Bỏ print ở hàm con

            all_true_labels_encoded_list.append(true_label_encoded)
            all_pred_labels_encoded_list.append(pred_label_encoded)
            detailed_results.append({
                'filename': fname, 'true_label': true_label_name,
                'predicted_label': pred_label_name, 'confidence': float(confidence),
                'is_correct': true_label_encoded == pred_label_encoded,
                'probabilities': avg_probs.tolist()
            })
        except Exception as e:
            print(f"  Error processing file {fname}: {e}")

    all_true_labels_encoded_np = np.array(all_true_labels_encoded_list)
    all_pred_labels_encoded_np = np.array(all_pred_labels_encoded_list)

    print("\n--- Test Set Performance Summary ---")
    if len(all_true_labels_encoded_np) == 0:
        print("No files were successfully tested or processed.")
        return

    accuracy = accuracy_score(all_true_labels_encoded_np, all_pred_labels_encoded_np)
    f1 = f1_score(all_true_labels_encoded_np, all_pred_labels_encoded_np, average='weighted', zero_division=0)

    print(f"Overall Test Accuracy: {accuracy:.4f}")
    print(f"Overall Weighted F1-Score: {f1:.4f}")

    print("\nClassification Report (Test Set):")
    try:
        report = classification_report(
            all_true_labels_encoded_np, all_pred_labels_encoded_np,
            target_names=actual_classes,
            labels=range(len(actual_classes)),
            zero_division=0
        )
        print(report)
    except ValueError as e:
         print(f"Error generating classification report: {e}")

    plot_custom_confusion_matrix(all_true_labels_encoded_np, all_pred_labels_encoded_np,
                                 actual_classes, filename=confusion_matrix_save_path)
    try:
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True) # Đảm bảo thư mục tồn tại
        with open(results_save_path, 'w') as f:
            # ... (ghi detailed_results)
            f.write("filename,true_label,predicted_label,confidence,is_correct,probabilities\n")
            for res in detailed_results:
                f.write(f"{res['filename']},{res['true_label']},{res['predicted_label']},{res['confidence']:.4f},{res['is_correct']},{res['probabilities']}\n")
        print(f"Detailed test results saved to {results_save_path}")
    except Exception as e:
         print(f"Error saving detailed test results: {e}")

    print("\n--- Saving Test Predictions and Labels for external evaluation ---")
    try:
        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(true_labels_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(class_names_save_path), exist_ok=True)

        with open(predictions_save_path, 'wb') as f:
            pickle.dump(all_pred_labels_encoded_np, f)
        print(f"Test predictions saved to {predictions_save_path}")
        # ... (lưu true_labels và class_names)
        with open(true_labels_save_path, 'wb') as f:
            pickle.dump(all_true_labels_encoded_np, f)
        print(f"Test true labels saved to {true_labels_save_path}")

        with open(class_names_save_path, 'wb') as f:
             pickle.dump(actual_classes, f)
        print(f"Class names saved to {class_names_save_path}")

    except Exception as e:
        print(f"Error saving test results arrays: {e}")
    # print("-" * 50)