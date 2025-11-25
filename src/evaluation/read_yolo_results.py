# read_yolo_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# Đường dẫn đến file results.csv (thay đổi nếu cần)
RESULTS_CSV_PATH = '/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/runs/train_vcam_youtube/v_cam_youtube_finetuned_model/results.csv'
# Thư mục lưu biểu đồ
PLOTS_SAVE_DIR = '/home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project/reports/figures/vcam_yolo_youtube_finetuned_training_analysis'

def analyze_yolo_results(csv_path, plots_dir):
    """
    Đọc file results.csv từ YOLO, hiển thị thông tin tóm tắt và vẽ biểu đồ.
    """
    if not os.path.exists(csv_path):
        print(f"LỖI: Không tìm thấy file results.csv tại: {csv_path}")
        return

    print(f"--- Phân tích kết quả từ: {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return

    # Loại bỏ khoảng trắng thừa ở tên cột (nếu có)
    df.columns = df.columns.str.strip()

    # Hiển thị 5 dòng đầu và 5 dòng cuối để xem qua dữ liệu
    print("\n--- 5 dòng đầu của dữ liệu ---")
    print(df.head())
    print("\n--- 5 dòng cuối của dữ liệu ---")
    print(df.tail())

    # Hiển thị thông tin cơ bản về các cột
    print("\n--- Thông tin cột ---")
    df.info()

    # Các cột quan trọng thường có
    epoch_col = 'epoch'
    train_box_loss_col = 'train/box_loss'
    train_cls_loss_col = 'train/cls_loss'
    # train_dfl_loss_col = 'train/dfl_loss' # Tùy phiên bản YOLO
    val_box_loss_col = 'val/box_loss'
    val_cls_loss_col = 'val/cls_loss'
    # val_dfl_loss_col = 'val/dfl_loss'   # Tùy phiên bản YOLO
    precision_col = 'metrics/precision(B)'
    recall_col = 'metrics/recall(B)'
    map50_col = 'metrics/mAP50(B)'
    map50_95_col = 'metrics/mAP50-95(B)'

    # Kiểm tra sự tồn tại của các cột chính
    required_metrics_cols = [epoch_col, map50_col, map50_95_col]
    required_loss_cols_train = [train_box_loss_col, train_cls_loss_col]
    required_loss_cols_val = [val_box_loss_col, val_cls_loss_col]

    missing_cols = [col for col in required_metrics_cols + required_loss_cols_train + required_loss_cols_val if col not in df.columns]
    if missing_cols:
        print(f"\nCẢNH BÁO: Thiếu các cột quan trọng sau trong file CSV: {', '.join(missing_cols)}")
        print("Không thể thực hiện phân tích đầy đủ.")
        # return # Có thể dừng ở đây hoặc cố gắng xử lý những gì có

    # Tìm epoch có mAP50-95 cao nhất trên tập validation
    if map50_95_col in df.columns:
        best_epoch_map50_95 = df[map50_95_col].idxmax() # Index của dòng có giá trị max
        best_map50_95_stats = df.loc[best_epoch_map50_95]
        print("\n--- Epoch có mAP50-95 (Validation) cao nhất ---")
        print(f"Epoch số: {int(best_map50_95_stats.get(epoch_col, -1)) + 1}") # Epoch thường bắt đầu từ 0 trong file
        print(best_map50_95_stats[[col for col in df.columns if 'val/' in col or 'metrics/' in col or col == epoch_col]])
    else:
        print(f"Cột '{map50_95_col}' không tồn tại, không thể tìm epoch tốt nhất dựa trên nó.")


    # Tìm epoch có mAP50 cao nhất trên tập validation
    if map50_col in df.columns:
        best_epoch_map50 = df[map50_col].idxmax()
        best_map50_stats = df.loc[best_epoch_map50]
        print("\n--- Epoch có mAP50 (Validation) cao nhất ---")
        print(f"Epoch số: {int(best_map50_stats.get(epoch_col, -1)) + 1}")
        print(best_map50_stats[[col for col in df.columns if 'val/' in col or 'metrics/' in col or col == epoch_col]])
    else:
        print(f"Cột '{map50_col}' không tồn tại.")


    # Thông tin ở epoch cuối cùng
    last_epoch_stats = df.iloc[-1]
    print("\n--- Thông số ở Epoch cuối cùng ---")
    print(last_epoch_stats)

    # Tạo thư mục lưu biểu đồ nếu chưa có
    os.makedirs(plots_dir, exist_ok=True)

    # Vẽ biểu đồ
    # 1. Biểu đồ Loss (Train vs Validation)
    plt.figure(figsize=(12, 6))
    # Tính tổng train loss (nếu có các thành phần loss)
    train_loss_cols_present = [col for col in required_loss_cols_train if col in df.columns]
    if train_loss_cols_present:
        df['train/total_loss'] = df[train_loss_cols_present].sum(axis=1)
        if epoch_col in df.columns:
            plt.plot(df[epoch_col], df['train/total_loss'], label='Train Total Loss')

    val_loss_cols_present = [col for col in required_loss_cols_val if col in df.columns]
    if val_loss_cols_present:
        df['val/total_loss'] = df[val_loss_cols_present].sum(axis=1)
        if epoch_col in df.columns:
            plt.plot(df[epoch_col], df['val/total_loss'], label='Validation Total Loss')
    
    if train_loss_cols_present or val_loss_cols_present:
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
        print(f"Biểu đồ Loss đã lưu vào: {os.path.join(plots_dir, 'loss_plot.png')}")
        plt.close()
    else:
        print("Không đủ dữ liệu loss để vẽ biểu đồ.")


    # 2. Biểu đồ mAP (mAP50 và mAP50-95 trên Validation)
    plt.figure(figsize=(12, 6))
    plot_map = False
    if map50_col in df.columns and epoch_col in df.columns:
        plt.plot(df[epoch_col], df[map50_col], label='Validation mAP@0.50')
        plot_map = True
    if map50_95_col in df.columns and epoch_col in df.columns:
        plt.plot(df[epoch_col], df[map50_95_col], label='Validation mAP@0.50-0.95')
        plot_map = True
    
    if plot_map:
        plt.title('Validation mAP Scores')
        plt.xlabel('Epoch')
        plt.ylabel('mAP Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'map_plot.png'))
        print(f"Biểu đồ mAP đã lưu vào: {os.path.join(plots_dir, 'map_plot.png')}")
        plt.close()
    else:
        print("Không đủ dữ liệu mAP để vẽ biểu đồ.")

    # 3. Biểu đồ Precision và Recall (Validation)
    plt.figure(figsize=(12, 6))
    plot_pr = False
    if precision_col in df.columns and epoch_col in df.columns:
        plt.plot(df[epoch_col], df[precision_col], label='Validation Precision')
        plot_pr = True
    if recall_col in df.columns and epoch_col in df.columns:
        plt.plot(df[epoch_col], df[recall_col], label='Validation Recall')
        plot_pr = True

    if plot_pr:
        plt.title('Validation Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'precision_recall_plot.png'))
        print(f"Biểu đồ Precision-Recall đã lưu vào: {os.path.join(plots_dir, 'precision_recall_plot.png')}")
        plt.close()
    else:
        print("Không đủ dữ liệu Precision/Recall để vẽ biểu đồ.")

    print("\n--- Phân tích hoàn tất ---")

if __name__ == '__main__':
    analyze_yolo_results(RESULTS_CSV_PATH, PLOTS_SAVE_DIR)