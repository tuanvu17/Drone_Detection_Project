# Drone_Detection_Project/src/evaluation/plotting_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_training_history(history, filename, title_prefix="", initial_epoch_offset=0): # Thêm initial_epoch_offset
    """
    Vẽ biểu đồ accuracy và loss từ đối tượng history của Keras.
    initial_epoch_offset: giá trị để cộng vào trục epoch (hữu ích cho Giai đoạn 2).
    """
    if history is None or not hasattr(history, 'history') or not history.history:
        print(f"Cảnh báo: Không có dữ liệu lịch sử huấn luyện để vẽ cho file: {filename}")
        return

    has_accuracy = 'accuracy' in history.history and 'val_accuracy' in history.history
    has_loss = 'loss' in history.history and 'val_loss' in history.history

    if not has_accuracy and not has_loss:
        print(f"Cảnh báo: Dữ liệu lịch sử huấn luyện không chứa 'accuracy'/'val_accuracy' hoặc 'loss'/'val_loss' cho file: {filename}")
        return

    num_plots = 0
    if has_accuracy: num_plots += 1
    if has_loss: num_plots += 1
    if num_plots == 0: return

    plt.figure(figsize=(6 * num_plots, 5))
    plot_idx = 1

    # Tạo mảng epochs cho trục x, có tính đến offset
    # history.epoch là list các chỉ số epoch đã chạy, ví dụ [0, 1, 2,... N-1] cho N epoch
    # Nếu history_stage2 được gọi với initial_epoch=20 và chạy 15 epoch,
    # thì history_stage2.epoch sẽ là [20, 21, ..., 34]
    # Tuy nhiên, history.history['accuracy'] vẫn sẽ là list có độ dài N.
    # Do đó, chúng ta cần tạo trục x dựa trên độ dài của history.history và offset.
    
    num_epochs_run = 0
    if has_accuracy:
        num_epochs_run = len(history.history['accuracy'])
    elif has_loss:
        num_epochs_run = len(history.history['loss'])

    epochs_range = range(initial_epoch_offset, initial_epoch_offset + num_epochs_run)


    if has_accuracy:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{title_prefix}Model Accuracy'.strip())
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        # Đặt giới hạn trục x để rõ ràng hơn
        if num_epochs_run > 0 : plt.xlim([initial_epoch_offset - 0.5, initial_epoch_offset + num_epochs_run - 0.5])
        plot_idx += 1

    if has_loss:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(epochs_range, history.history['loss'], label='Train Loss')
        plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
        plt.title(f'{title_prefix}Model Loss'.strip())
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        if num_epochs_run > 0 : plt.xlim([initial_epoch_offset - 0.5, initial_epoch_offset + num_epochs_run - 0.5])


    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"Biểu đồ lịch sử huấn luyện đã được lưu vào: {filename}")
    except Exception as e_save:
        print(f"Lỗi khi lưu biểu đồ lịch sử huấn luyện: {e_save}")
    plt.close()


def plot_custom_confusion_matrix(y_true, y_pred_classes, classes, filename, title_prefix=""): # Thêm title_prefix
    # ... (Giữ nguyên hàm này, chỉ thêm title_prefix vào title của plot) ...
    if len(y_true) == 0 or len(y_pred_classes) == 0:
         print(f"Cảnh báo: Không thể vẽ ma trận nhầm lẫn cho {filename} do thiếu dữ liệu nhãn thực tế hoặc dự đoán.")
         return
    if not classes:
         unique_labels = np.unique(np.concatenate((y_true, y_pred_classes)))
         cm_labels = unique_labels
         cm_xticklabels = [str(l) for l in unique_labels]
         cm_yticklabels = [str(l) for l in unique_labels]
         print(f"Cảnh báo: Danh sách tên lớp rỗng cho {filename}. Sử dụng giá trị nhãn duy nhất.")
    else:
        cm_labels = range(len(classes))
        cm_xticklabels = classes
        cm_yticklabels = classes

    try:
        cm = confusion_matrix(y_true, y_pred_classes, labels=cm_labels)
        plt.figure(figsize=(max(8, int(len(cm_xticklabels)*0.8) ), max(6, int(len(cm_yticklabels)*0.6))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cm_xticklabels, yticklabels=cm_yticklabels)
        plt.title(f'{title_prefix}Confusion Matrix'.strip()) # Thêm title_prefix
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"Ma trận nhầm lẫn đã được lưu vào: {filename}")
        plt.close()
    except Exception as e:
        print(f"Lỗi khi vẽ ma trận nhầm lẫn cho {filename}: {e}")

# Drone_Detection_Project/src/evaluation/plotting_utils.py
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
# import numpy as np # Thêm import numpy nếu cần xử lý thêm
# import os # Thêm import os nếu cần xử lý đường dẫn

def plot_custom_confusion_matrix(y_true, y_pred_classes, classes, filename, title=None):
    """
    Vẽ và lưu ma trận nhầm lẫn.

    Args:
        y_true (array-like): Nhãn thực tế (dạng số nguyên).
        y_pred_classes (array-like): Nhãn dự đoán (dạng số nguyên).
        classes (list): Danh sách tên các lớp theo đúng thứ tự của nhãn số.
        filename (str): Đường dẫn đầy đủ để lưu file ảnh ma trận nhầm lẫn.
        title (str, optional): Tiêu đề cho biểu đồ. Mặc định là 'Confusion Matrix'.
    """
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_pred_classes, (list, np.ndarray)):
        print(f"Cảnh báo cho {filename}: y_true hoặc y_pred_classes không phải là list hoặc numpy array.")
        return
    if len(y_true) == 0 or len(y_pred_classes) == 0:
        print(f"Không thể vẽ CM cho {filename}: Không có nhãn thực tế hoặc dự đoán.")
        return
    if not classes or not isinstance(classes, (list, tuple)) or not all(isinstance(c, str) for c in classes):
        print(f"Không thể vẽ CM cho {filename}: Danh sách 'classes' không hợp lệ (phải là list/tuple các chuỗi).")
        return
    if len(y_true) != len(y_pred_classes):
        print(f"Không thể vẽ CM cho {filename}: Số lượng nhãn không khớp ({len(y_true)} vs {len(y_pred_classes)}).")
        return

    try:
        # Xác định các nhãn duy nhất có mặt trong y_true và y_pred_classes
        # và đảm bảo chúng là các chỉ số hợp lệ cho danh sách 'classes'
        unique_true_labels = set(y_true)
        unique_pred_labels = set(y_pred_classes)
        all_present_labels = sorted(list(unique_true_labels | unique_pred_labels))

        # Lọc ra các labels mà có trong `classes`
        # labels_for_cm: là các ID lớp (0, 1, 2,...) sẽ được hiển thị trên ma trận
        # class_names_for_display: là tên lớp tương ứng với labels_for_cm
        labels_for_cm = [l for l in all_present_labels if l < len(classes)]
        if not labels_for_cm:
            # Nếu các ID lớp trong y_true/y_pred không khớp với độ dài của `classes`
            # hoặc không có nhãn nào hợp lệ, thử dùng tất cả các ID có thể có từ 0 đến len(classes)-1
            # Điều này hữu ích khi một số lớp có thể không xuất hiện trong y_true/y_pred cụ thể.
            labels_for_cm = list(range(len(classes)))
            class_names_for_display = list(classes)
            # print(f"Cảnh báo cho {filename}: Không có nhãn nào khớp với `classes` trong y_true/y_pred. Sử dụng tất cả các lớp được định nghĩa.")
        else:
            class_names_for_display = [classes[i] for i in labels_for_cm]


        # Tính ma trận nhầm lẫn chỉ với các nhãn hợp lệ này
        cm = sklearn_confusion_matrix(y_true, y_pred_classes, labels=labels_for_cm)

        plt.figure(figsize=(max(8, len(class_names_for_display) * 1.2), max(6, len(class_names_for_display) * 0.9)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names_for_display,
                    yticklabels=class_names_for_display,
                    annot_kws={"size": 10})

        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title('Confusion Matrix', fontsize=14)

        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"Ma trận nhầm lẫn đã lưu vào: {filename}")
        plt.close()

    except Exception as e:
        print(f"Lỗi khi vẽ ma trận nhầm lẫn cho {filename}: {e}")
        import traceback
        traceback.print_exc()
        if plt.get_fignums(): # Kiểm tra xem có figure nào đang mở không
            plt.close() # Đảm bảo đóng plot nếu có lỗi

# Bạn có thể thêm các hàm vẽ biểu đồ khác vào đây nếu cần
# Ví dụ: plot_training_history, plot_precision_recall_curve, etc.