# Drone_Detection_Project/src/evaluation/plotting_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np # Cần cho labels trong confusion_matrix nếu classes rỗng

def plot_training_history(history, filename):
    if history is None or not hasattr(history, 'history') or not history.history:
        print("No training history to plot.")
        return
    try:
        plt.figure(figsize=(12, 5))
        # Accuracy
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
        else:
            print("Warning: 'accuracy' or 'val_accuracy' not found in history.")

        # Loss
        if 'loss' in history.history and 'val_loss' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
        else:
            print("Warning: 'loss' or 'val_loss' not found in history.")

        plt.tight_layout()
        plt.savefig(filename)
        print(f"Training history plot saved to {filename}")
        plt.close()
    except Exception as e:
        print(f"Error plotting training history: {e}")

def plot_custom_confusion_matrix(y_true, y_pred_classes, classes, filename):
    if len(y_true) == 0 or len(y_pred_classes) == 0:
         print("Cannot plot confusion matrix: No true or predicted labels provided.")
         return
    if not classes: # Nếu list classes rỗng
         # Tạo labels mặc định dựa trên số lượng lớp duy nhất trong y_true hoặc y_pred
         unique_labels = np.unique(np.concatenate((y_true, y_pred_classes)))
         cm_labels = unique_labels
         cm_xticklabels = [str(l) for l in unique_labels]
         cm_yticklabels = [str(l) for l in unique_labels]
         print("Warning: Class names list is empty. Using unique label values for confusion matrix.")
    else:
        cm_labels = range(len(classes))
        cm_xticklabels = classes
        cm_yticklabels = classes

    try:
        cm = confusion_matrix(y_true, y_pred_classes, labels=cm_labels)
        plt.figure(figsize=(max(8, len(cm_xticklabels)), max(6, len(cm_yticklabels)))) # Điều chỉnh kích thước
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cm_xticklabels, yticklabels=cm_yticklabels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Confusion matrix plot saved to {filename}")
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")