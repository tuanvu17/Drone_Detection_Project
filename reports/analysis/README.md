# Hướng Dẫn Tạo Báo Cáo Kết Quả Audio

## Mục Đích

Script này tạo báo cáo kết quả và phân tích cho các mô hình Audio theo yêu cầu:

- **4.4.1**: Kết quả trên Bộ Dữ liệu Gốc (best_audio_model.keras)
- **4.4.2**: Kết quả trên Bộ Dữ liệu YouTube Fine-tune (best_audio_youtube_finetuned_model.keras)

## Cách Chạy

```bash
cd /home/tuanvu17/mydocuments/ths/luanvan/Drone_Detection_Project
python -m src.main_generate_audio_results
```

## Kết Quả

Sau khi chạy, các file sau sẽ được tạo trong `reports/analysis/`:

### 4.4.1. Bộ Dữ Liệu Gốc
- `audio_original/classification_report_original.txt` - Classification Report
- `audio_original/confusion_matrix_original.png` - Confusion Matrix

### 4.4.2. Bộ Dữ Liệu YouTube
- `audio_youtube/classification_report_youtube.txt` - Classification Report
- `audio_youtube/confusion_matrix_youtube.png` - Confusion Matrix

### Báo Cáo Tổng Hợp
- `audio_results_analysis.md` - Báo cáo phân tích tổng hợp (Markdown format)

## Nội Dung Báo Cáo

Báo cáo bao gồm:

1. **Classification Report**: 
   - Precision, Recall, F1-score cho từng lớp
   - Accuracy tổng thể
   - Macro và Weighted averages

2. **Confusion Matrix**: 
   - Ma trận nhầm lẫn dạng hình ảnh
   - Hiển thị số lượng dự đoán đúng/sai cho từng lớp

3. **Phân Tích Hiệu Năng**:
   - Phân tích khả năng phân biệt các lớp của đặc trưng MFCC
   - So sánh hiệu năng giữa 2 bộ dữ liệu
   - Đánh giá tính tổng quát của mô hình

## Yêu Cầu

- Các model đã được huấn luyện và lưu tại:
  - `models/audio_classifier/best_audio_model.keras`
  - `models/audio_classifier_youtube_finetuned/best_audio_youtube_finetuned_model.keras`
- Test data đã được chuẩn bị
- Các file scaler, label_encoder, max_len đã được lưu

